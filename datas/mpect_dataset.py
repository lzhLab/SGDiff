"""
MECT数据集
——train:498
    ——d.nii
    ——j.nii
    ——y.nii
——test:62
——val:62
"""
import os
import matplotlib.pyplot as plt
from monai import transforms
from torch.utils.data import Dataset
from monai.data import Dataset as MonaiDataset
from datas.data_util import TemporalRandomCrop, get_sobel


def get_ct_dataset(data_path, transform):
    datas = []
    name = ""
    for i in sorted(os.listdir(data_path), key=lambda x: int(x)):
        nii_path = os.path.join(data_path, i)
        for j in sorted(os.listdir(nii_path)):
            niifile = os.path.join(nii_path, j)
            flag = j.split('.')[0]
            if flag == "d":
                name = "ART"
            elif flag == "j":
                name = "PV"
            elif flag == "y":
                name = "DL"
            datas.append({"image": niifile, "class_name": name, })
    return MonaiDataset(data=datas, transform=transform)

# 训练transforms
train_transforms = transforms.Compose([
    # 加载图像
    transforms.LoadImaged(keys=["image"], allow_missing_keys=True),
    # 增加维度
    transforms.EnsureChannelFirstD(keys=["image"]),
    # 重采样
    transforms.Spacingd(keys=["image"], pixdim=(1.5, 1.5, 5.0)),
    # 重定向轴
    transforms.Orientationd(keys=["image"], axcodes="SPL"),
    TemporalRandomCrop(keys=["image"], size=16),
    transforms.SpatialPadd(keys=["image"], spatial_size=(16, 256, 256), method="symmetric", mode="constant"),
    # 中心裁剪
    transforms.CenterSpatialCropd(keys=["image"], roi_size=(16, 256, 256)),
    # 图像缩放到-1~1
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255, b_min=-1.0, b_max=1.0),
    transforms.ToTensor(),
])

# 测试transforms
test_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"], allow_missing_keys=True),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.Spacingd(keys=["image"], pixdim=(1.5, 1.5, 5.0)),
        transforms.Orientationd(keys=["image"], axcodes="SPL"),
        transforms.SpatialPadd(keys=["image"], spatial_size=(16, 256, 256), method="symmetric", mode="constant"),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=(16, 256, 256)),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255, b_min=-1.0, b_max=1.0),
        transforms.ToTensor(),
    ]
)

# 基本数据集
class BaseDataset(Dataset):
    def __init__(self, data_path, transform, use_cond=False):
        """
        :param data_path: 数据路径
        :param transform: 所用的transform
        :param use_cond:  条件类型
        """
        # 获取ct数据
        self.ct_data = get_ct_dataset(data_path, transform=transform)
        self.use_cond = use_cond

    def __len__(self):
        # 返回数据长度
        return len(self.ct_data)

    def __getitem__(self, i):
        datasets = self.ct_data[i]
        if self.use_cond == "sobel":
            datasets["sobel"] = get_sobel(datasets["image"])
        return datasets


# 训练数据集
class TrainDataset(BaseDataset):
    def __init__(self, data_path, transform=train_transforms, use_cond=None, **kwargs):
        super().__init__(data_path=data_path, transform=transform, use_cond=use_cond)


# 验证数据集
class ValDataset(BaseDataset):
    def __init__(self, data_path, transform=test_transforms, use_cond=None, **kwargs):
        super().__init__(data_path=data_path, transform=transform, use_cond=use_cond)


# 测试数据集
class TestDataset(BaseDataset):
    def __init__(self, data_path, transform=test_transforms, use_cond=None, **kwargs):
        super().__init__(data_path=data_path, transform=transform, use_cond=use_cond)


if __name__ == "__main__":
    # A800 数据集地址
    # data_dir = "/home/zhaoliang/data/DATASET/MPECT"
    # v100 数据集地址
    data_dir = "/data1/lzp/data/MPECT/test"
    # 获取训练数据
    train_dataset = TestDataset(data_path=data_dir, use_cond="sobel")
    dataset = train_dataset[0]
    image = (dataset["image"] + 1) * 127.5
    img = image.squeeze(0)
    img = img[0, :, :]
    plt.imshow(img, cmap="gray")
    plt.show()
