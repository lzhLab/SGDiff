import os
import sys
import torch
import argparse
import warnings
sys.path.append('../')
warnings.filterwarnings("ignore")
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from sample_util import save_args, load_model, all_testdata_sample



def get_parser():
    parser = argparse.ArgumentParser()
    # basic args
    parser.add_argument("--save_dir", type=str, help="结果保存地址",
                        # default="/home/lzp/results/Ours/vqgan"
                        default="/data1/lzp/results/Ours/vae")
    parser.add_argument("--dataset_name", type=str, help="数据集名",
                        default="MPECT")
    parser.add_argument("--gpu_id", type=int, help="选择一个特定的gpu",
                        default=2)
    parser.add_argument("--seed", type=int, help="固定一个随机种子",
                        default=597, )
    parser.add_argument("--batch_size", type=int, help="video batch size for sampling",
                        default=1)
    parser.add_argument("--config_path", type=str, help="模型配置文件路径",
                        # default="/home/lzhao/model_save/vqgan/logs/vqgan_v1/configs/2024-08-30:18-26-39-project.yaml"
                        # default="/home/lzp/model_save/vqgan/2024-08-30:18-26-39-project.yaml"
                        default="/home/lzp/project/SGDiff/config/vae/pretrain_vae.yaml")
    # basic args
    parser.add_argument("--ckpt_path", type=str, help="模型保存路径",
                        # default="/home/lzhao/model_save/vqgan/logs/vqgan_v1/checkpoints/epoch=0318-step=119306.ckpt",
                        # default="/home/lzp/model_save/vqgan/epoch=0318-step=119306.ckpt",
                        # default="/home/lzp/model_save/vqgan/epoch=0342-step=128282.ckpt",
                        # default="/home/lzp/model_save/vqgan/epoch=0372-step=139502.ckpt",
                        default="/home/lzp/model_save/vae/vae_144_e=0120_loss=0.03525.ckpt")
    parser.add_argument("--data_dir", type=str, help="数据地址",
                        default="/data1/lzp/data/MPECT/test",
                        # default="/home/lzp/data/VDMP/test"
                        # default="/home/lzp/data/VDMP/sample"
                        )
    return parser


if __name__ == "__main__":
    # 配置参数
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    # 保存地址
    save_dir = os.path.join(opt.save_dir, opt.dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    # 保持采样参数
    # save_args(save_dir, opt)

    torch.cuda.set_device(opt.gpu_id)
    device = torch.device('cuda:{}'.format(opt.gpu_id))
    # 配置随机种子
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # load & merge config
    config = OmegaConf.load(opt.config_path)
    # 从命令行参数中创建一个配置对象cil
    cli = OmegaConf.from_dotlist(unknown)
    # 将命令行参数中的配置对象cil与配置文件中的配置对象config进行合并，生成一个新的配置对象
    config = OmegaConf.merge(config, cli)
    # print("config: \n", config)

    # get model
    model, _, _ = load_model(config, opt.ckpt_path)
    model = model.to(device)

    if opt.dataset_name == "MPECT":
        from datas.mpect_dataset import TestDataset

    test_dataset = TestDataset(data_path=opt.data_dir)
    print(f"Creat Dataset: [{opt.dataset_name}]")
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    # 整个测试数据集进行采样生成, phase:指定期相——0:动脉期, 1:静脉期, 2:延迟期
    all_testdata_sample(model, test_dataloader, device, save_dir, phase=0)
