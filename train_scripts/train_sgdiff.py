"""
SGDiff训练
"""
import os
import math
import sys
import torch
import imageio
sys.path.append("..")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import torchvision
import warnings
warnings.filterwarnings("ignore")
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from datas import get_dataset
from omegaconf import OmegaConf
from model.latte import get_models
from torch.utils.data import DataLoader
from diffusion import create_diffusion
from diffusers.optimization import get_scheduler
from train_util import \
    setup_distributed, get_args, create_logger, \
    create_tensorboard, Load_VAE, requires_grad, \
    update_ema, clip_grad_norm_, latent_space, \
    save_img_grid

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ["LOCAL_RANK"] = '0'
os.environ["SLURM_NTASKS"] = '1'


def get_parser():
    parser = argparse.ArgumentParser()
    # basic args
    parser.add_argument("--version", type=str, default="v1", help="训练版本")
    parser.add_argument("--port", type=int, default=6033, help="使用端口号")
    parser.add_argument("--seed", type=int, default=1234, help="设置随机种子")
    parser.add_argument("--results_dir", type=str, help="结果保存地址",
                        # default="/data1/lzp/results/Ours/SGDiff"
                        default="/home/zhaoliang/results/Ours/SGDiff"
                        )
    parser.add_argument("--config_path", type=str, help="配置文件地址",
                        # default="/home/lzp/project/SGDiff/config/sgdiff/train/mpect_v1.yaml"
                        default="/home/zhaoliang/student/lzp/SGDiff/config/sgdiff/train/mpect_v1.yaml"
                        )
    parser.add_argument("--pretrained", type=str, help="预训练模型地址",)
    parser.add_argument("--resume_from_checkpoint", type=str)
    parser.add_argument("--data_path", type=str, help="数据地址",
                        # default="/data1/lzp/data/MPECT"
                        default="/home/zhaoliang/data/DATASET/MPECT"
                        )
    # VAE args
    parser.add_argument("--vae_ckpt_path", type=str,
                        # default="/home/lzp/model_save/vqgan/epoch=0342-step=128282.ckpt",
                        # default="/home/lzp/model_save/vae/vae_144_e=0120_loss=0.03525.ckpt",
                        default="/home/zhaoliang/student/lzp/model_save/prtrain_model/vae_144_e=0120_loss=0.03525.ckpt"
                        )
    parser.add_argument("--vae_config", type=str,
                        # default="/home/lzp/model_save/vqgan/2024-08-30:18-26-39-project.yaml"
                        # default="/home/lzp/project/MALDM/config/vae/pretrain_vae.yaml"
                        default="/home/zhaoliang/student/lzp/MALDM/config/vae/pretrain_vae.yaml"
                        )

    # train args
    parser.add_argument("--phase", type=str, default="train", help="train or test")
    parser.add_argument("--local_batch_size", type=int, default=4, help="batch size大小")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=1000000, help="最大训练步骤")
    parser.add_argument("--start_clip_iter", type=int, default=100000)
    parser.add_argument("--clip_max_norm", type=float, default=0.1)
    # 调试用例: 1 实际训练用50
    parser.add_argument("--log_every", type=int, default=100, help="每log_every步检查损失")
    # 测试用例: 1 实际训练用
    parser.add_argument("--ckpt_every", type=int, default=5000, help="每ckpt_every步保持模型并采样图像")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    # low VRAM and speed up training
    parser.add_argument("--use_compile", type=str, )

    return parser


if __name__ == "__main__":
    # 获取基本参数
    parser = get_parser()
    args = parser.parse_args()
    # 获取config配置文件参数
    file_args = OmegaConf.load(args.config_path)
    args = get_args(file_args, args)
    assert torch.cuda.is_available(), "Training currently requires at least one GPU"
    os.environ['MASTER_PORT'] = str(args.port)
    # Setup DDP:
    setup_distributed()
    # 获得训练用rank, local_rank, world_size, 使用gpu设备号为local_rank
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # local_rank = rank
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device("cuda", local_rank)
    # 设置随机种子
    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={world_size}.")

    # Setup an experiment folder:
    if rank == 0:
        # 创建结果文件夹(包括有实验子文件夹)
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        # 模型名, e.g., Latte-XL/2 --> Latte-XL-2 (for naming folders)
        model_string_name = args.model.split("-")[0]
        # Create an experiment folder
        experiment_dir = f"{args.results_dir}/{args.version}-{experiment_index:03d}-{model_string_name}-{args.dataset}"
        # Stores saved model checkpoints
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        # saved 训练过程中可视化结果保存地址
        images_dir = f"{experiment_dir}/images"
        os.makedirs(images_dir, exist_ok=True)
        # create logger and tensorboard
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        OmegaConf.save(vars(args), os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
        tb_writer = None

    # Create model
    assert args.image_size % 4 == 0, "Image size must be divisible by 8 (for the VAE encoder)"
    sample_size = args.image_size // 4
    args.latent_size = sample_size
    model = get_models(args).to(device)
    # default: 1000 steps, linear noise schedule
    diffusion = create_diffusion(timestep_respacing="")

    # 加载预训练vae
    vae_model, _, _ = Load_VAE(config=args.vae_config, ckpt_path=args.vae_ckpt_path, gpu_id=local_rank)
    vae_model = vae_model.to(device)

    # use pretrained model?
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            logger.info('Using ema ckpt!')
            checkpoint = checkpoint["ema"]
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {}
            for k, v in checkpoint.items():
                if k in model_dict:
                    pretrained_dict[k] = v
                else:
                    logger.info('Ignoring: {}'.format(k))
            logger.info('Successfully Load {}% original pretrained model weights '.format(
                len(pretrained_dict) / len(checkpoint.items()) * 100))
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            logger.info('Successfully load model at {}!'.format(args.pretrained))

    if args.use_compile:
        model = torch.compile(model)

    # 请注意，参数初始化是在 Latte 构造函数中完成的
    # 创建模型的 EMA，供训练后使用
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    logger.info(f"{args.model} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Freeze vae
    vae_model.requires_grad_(False)

    # 加载数据
    train_dataset = get_dataset(args, args.phase)
    val_dataset = get_dataset(args, "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.local_batch_size),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Train dataset contains {len(train_dataset):,} videos ({args.data_path})")
    logger.info(f"Val dataset contains {len(val_dataset):,} videos ({args.data_path})")

    # Scheduler
    # ConstantLR:在total_iters 轮内将optimizer里面指定的学习率乘以factor, total_iters轮外恢复原学习率
    lr_scheduler = get_scheduler(
        name="constant",   # 学习率类型
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    start_time = time()

    # 我们需要重新计算我们的总训练步数，因为训练数据加载器的大小可能已经改变了。
    num_update_steps_per_epoch = math.ceil(len(train_loader))
    # 之后我们重新计算训练周期的数量
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # TODO, need to checkout
        # Get the most recent checkpoint
        dirs = os.listdir(os.path.join(experiment_dir, 'checkpoints'))
        dirs = [d for d in dirs if d.endswith("pt")]
        dirs = sorted(dirs, key=lambda x: int(x.split(".")[0]))
        path = dirs[-1]
        logger.info(f"Resuming from checkpoint {path}")
        model.load_state(os.path.join(dirs, path))
        train_steps = int(path.split(".")[0])

        first_epoch = train_steps // num_update_steps_per_epoch
        resume_step = train_steps % num_update_steps_per_epoch

    if args.pretrained:
        train_steps = int(args.pretrained.split("/")[-1].split('.')[0])

    for epoch in range(first_epoch, num_train_epochs):
        # 训练框架
        for step, video_data in enumerate(train_loader):
            model.train()
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue
            # 获取训练数据x_0
            x = video_data['image'].to(device, non_blocking=True)
            x = torch.tensor(x, dtype=torch.float32, device=device)
            video_name = video_data['class_name']
            # 通过预训练编码器: x_0 -> z_0
            with torch.no_grad():
                z = latent_space(vae_model, x)
                z = rearrange(z, "b c f h w -> b f c h w")
            # 获取类别条件（如果extras == 2）
            if args.extras == 2:
                if isinstance(video_name, list):
                    video_labels = []
                    for i in range(len(video_name)):
                        if video_name[i] == "ART":
                            video_labels.append(0)
                        elif video_name[i] == "PV":
                            video_labels.append(1)
                        else:
                            video_labels.append(2)
                    video_label = torch.tensor(video_labels, dtype=torch.long, device=device)
                    model_kwargs = dict(y=video_label)
            else:
                model_kwargs = dict(y=None)
            # 随机采样时间步
            t = torch.randint(0, diffusion.num_timesteps, (z.shape[0],), device=device)
            # 计算loss
            loss_dict = diffusion.training_losses(model, z, t, model_kwargs)
            loss = loss_dict["loss"].mean() / args.gradient_accumulation_steps
            loss.backward()
            # if train_steps >= start_clip_iter, will clip gradient
            if train_steps < args.start_clip_iter:
                gradient_norm = clip_grad_norm_(model.parameters(), args.clip_max_norm, clip_grad=False)
            else:
                gradient_norm = clip_grad_norm_(model.parameters(), args.clip_max_norm, clip_grad=True)

            lr_scheduler.step()
            if train_steps % args.gradient_accumulation_steps == 0 and train_steps > 0:
                opt.step()
                opt.zero_grad()
                update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # 测量训练速度:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # 减少所有流程的损失记录:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                # dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                # logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                logger.info(
                    f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # 仅适用于DDP
                # write_tensorboard(tb_writer, 'Train Loss', avg_loss, train_steps)
                # write_tensorboard(tb_writer, 'Gradient Norm', gradient_norm, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save Latte checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    # sampling
                    model.eval()
                    with torch.no_grad():
                        # 随机采样潜在噪声
                        z = torch.randn(1, args.num_frames, 4,  args.latent_size,  args.latent_size, device=device)
                        # 使用cfg进行采样, 有类别条件采样的情况下
                        z = torch.cat([z, z], 0)
                        # y = torch.tensor([args.class_phase], device=device)
                        y = torch.randint(0, args.num_classes, (1,), device=device)
                        y_null = torch.tensor([args.num_classes] * 1, device=device)
                        y = torch.cat([y, y_null], dim=0)
                        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                        sample_fn = model.forward_with_cfg
                        # 使用ddim采样方式
                        samples = diffusion.p_sample_loop(
                            sample_fn, z.shape, z, clip_denoised=False,
                            model_kwargs=model_kwargs, progress=True, device=device
                        )
                        print(samples.shape)
                        b, f, c, h, w = samples.shape
                        samples = rearrange(samples, 'b f c h w -> b c f h w')
                        samples = vae_model.decode(samples / 0.18215)
                        samples = rearrange(samples, 'b c f h w -> b f c h w', b=b)
                        samples_dir = os.path.join(images_dir, f"TS:{train_steps}-E:{epoch}-Label:{y[0]}-Loss:{avg_loss:.4f}")
                        if not os.path.exists(samples_dir):
                            os.makedirs(samples_dir)
                        video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(
                            dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
                        video_save_path = os.path.join(samples_dir, 'sample' + '.mp4')
                        imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
                        print('save path {}'.format(samples_dir))
                        video_ = video_.permute(0, 3, 1, 2)
                        grid = torchvision.utils.make_grid(video_, nrow=video_.shape[0])
                        path = os.path.join(samples_dir, 'sample' + '.png')
                        save_img_grid(grid, path)

                    # checkpoint
                    checkpoint = {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        # "opt": opt.state_dict(),
                        # "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")


                    # evalution
                # dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    #  使用评估模式下的 ema（或模型）进行任何采样/FID 计算等 ...

    logger.info("Done!")
    # cleanup()

