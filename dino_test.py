import model.dino.vision_transformer as vits
import torch
import requests
from io import BytesIO
from PIL import Image
from torchvision import transforms as pth_transforms

dino_weight = "/home/zhaoliang/student/lzp/model_save/prtrain_model/dino_vitbase8_pretrain.pth"

device = "cuda:0"


def load_model(device, pretrained_path):
    model = vits.__dict__["vit_base"](
        patch_size=8, num_classes=0
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    state_dict = torch.load(pretrained_path, map_location="cpu")

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(
        "Pretrained weights found at {} and loaded with msg: {}".format(
            pretrained_path, msg
        )
    )

    return model

dino = load_model(device, dino_weight)

# 打开图像
response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
img = Image.open(BytesIO(response.content))
img = img.convert('RGB')

transform = pth_transforms.Compose([
        pth_transforms.Resize((480, 480)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
img = transform(img)
print(img.shape)
