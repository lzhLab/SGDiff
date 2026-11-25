import os
import yaml
import torch
import time
from tqdm import tqdm
from utils import instantiate_from_config
from torchvision.utils import save_image

# -1~1 -> 0~255
to_range_255 = lambda x: (x + 1) * 127.5

# 保存配置参数
def save_args(save_dir, args):
    fpath = os.path.join(save_dir, "sampling_args.yaml")
    with open(fpath, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)


# 加载模型
def load_model(config, ckpt_path, gpu_id=None):
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


# 用于sample_vae.py 整个测试数据集采样
def all_testdata_sample(model, test_dataloader, device, save_dir, phase=0):
    start = time.time()
    for i, data in enumerate(tqdm(test_dataloader)):
        # i:0,3,6,9,.....—— 动脉期, i % 3
        # i:1,4,7,10,.....—— 静脉期, (i + 1) % 3
        # i:2,5,8,11,.....—— 延迟期, (i + 2) % 3
        # 只生成选中的对应期相
        if (i + phase) % 3 != 0:
            continue
        imgs = data["image"].to(device)
        with torch.no_grad():
            output, _ = model(imgs)
        output = to_range_255(output.squeeze().clamp(-1., 1.))
        fake_dir = os.path.join(save_dir, f"{((i + phase) // 3)}")
        if not os.path.exists(fake_dir):
            os.makedirs(fake_dir)
        for j in range(output.shape[0]):
            save_image(output[j, :, :], os.path.join(fake_dir, f"samples_{j}_a2b.png"), normalize=True)
    print("Finish sampling!")
    print(f"Run time = {(time.time() - start):.2f} seconds")
    print("Save dir :", save_dir)