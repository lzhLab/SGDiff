import torch
import os
import subprocess
import argparse
import logging
import importlib
import numpy as np
from PIL import Image
from torch import inf
from omegaconf import OmegaConf
from typing import Union, Iterable
import torch.distributed as dist
from einops import rearrange
from collections import OrderedDict
from model.latte import get_models
from torch.utils.tensorboard import SummaryWriter
from autoencoder.modules.distributions import DiagonalGaussianDistribution

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def save_img_grid(grid, path):
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy().astype(np.uint8)
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    Image.fromarray(grid).save(path)


# 将文件配件参数与命令配置参数结合
def get_args(file_args, basic_args):
    # 将命令参数转成字典
    args_default = vars(basic_args)
    # 将文件参数更新到命令参数字典
    args_default.update(file_args)
    args = argparse.Namespace(**args_default)
    return args


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            # os.environ["MASTER_PORT"] = "29566"
            os.environ["MASTER_PORT"] = str(29567 + num_gpus)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    # torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


#################################################################################
#                             Training Logger                                   #
#################################################################################

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            # format='[\033[34m%(asctime)s\033[0m] %(message)s',
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)

    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def create_tensorboard(tensorboard_dir):
    """
    Create a tensorboard that saves losses.
    """
    if dist.get_rank() == 0:  # real tensorboard
        # tensorboard
        writer = SummaryWriter(tensorboard_dir)

    return writer


#################################################################################
#                                  Load Model                                   #
#################################################################################
def Load_VAE(config, ckpt_path, gpu_id=None):
    # load & merge config
    config = OmegaConf.load(config)
    print(f"Loading model from {ckpt_path}")
    # load sd
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    try:
        global_step = pl_sd["global_step"]
        epoch = pl_sd['epoch']
    except:
        global_step = -1
        epoch = -1

    # load sd to model
    try:
        sd = pl_sd['state_dict']
    except:
        sd = pl_sd
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=True)

    # move to device & eval
    if gpu_id is not None:
        model.to(f"cuda:{gpu_id}")
    else:
        model.cuda()
    model.eval()
    return model, global_step, epoch


def get_obj_from_str(string, reload=False):
    # module是py文件的路径，cls是该py文件当中的模型module类
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # 使用importlib.import_module函数导入指定的模块
    # 使用getattr函数获取导入的模块中的指定类对象
    # 返回获取到的类对象
    return getattr(importlib.import_module(module, package=None), cls)


# 利用当前配置现的target和params属性来创建相应的对象
def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    # 初始化模型并传入config中的参数
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                      EMA Update/ DDP Training Utils                           #
#################################################################################
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


#################################################################################
#                                   VAE Encoder                                 #
#################################################################################
# 从高斯分布中采样, 并且缩放0.18215倍使得隐变量方差接近1
def get_first_stage_encoding(encoder_posterior, noise=None, scale_factor=0.18215, shift_factor=0.0):
    # vae
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        z = encoder_posterior.sample(noise=noise)
    # vq-model
    elif isinstance(encoder_posterior, torch.Tensor):
        z = encoder_posterior
    else:
        raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
    z = scale_factor * (z + shift_factor)
    return z


def clip_grad_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, clip_grad = True) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)

    if clip_grad:
        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(
                f'The total norm of order {norm_type} for gradients from '
                '`parameters` is non-finite, so it cannot be clipped. To disable '
                'this error and scale the gradients by the non-finite norm anyway, '
                'set `error_if_nonfinite=False`')
        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))
        # gradient_cliped = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    return total_norm

# 像素空间->潜在空间
def latent_space(vae_model, x):
    # Map input images to latent space + normalize latents:
    b, _, _, _, _ = x.shape
    # 将x转换为连续内存格式，并将其类型转换为float
    x = x.to(memory_format=torch.contiguous_format).float()
    z = vae_model.encode(x)
    z = get_first_stage_encoding(z).detach()
    return z

