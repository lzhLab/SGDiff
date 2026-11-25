import os.path
import random
import numpy as np
from einops import rearrange
from monai import transforms
from monai.transforms.transform import MapTransform
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from typing import Mapping, Hashable, Dict
from .mpect_dataset import TrainDataset, TestDataset


class TemporalRandomCrop(MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 size):
        super(TemporalRandomCrop, self).__init__(keys)
        self.size = size

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = rearrange(d[key], 'c t h w -> t c h w')
            total_frames = d[key].shape[0]
            rand_end = max(0, total_frames - self.size - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.size, total_frames)
            frame_indice = np.linspace(begin_index, end_index - 1, self.size, dtype=int)
            d[key] = d[key][frame_indice]
            d[key] = rearrange(d[key], 't c h w -> c t h w')
        return d


def get_dataset(args, phase):
    if args.dataset == 'MPECT':
        if phase == "train":
            mpect_transforms = transforms.Compose([
                # 加载图像
                transforms.LoadImaged(keys=["image"], allow_missing_keys=True),
                # 增加维度
                transforms.EnsureChannelFirstD(keys=["image"]),
                # 重采样
                transforms.Spacingd(keys=["image"], pixdim=(1.5, 1.5, 5.0)),
                # 重定向轴
                transforms.Orientationd(keys=["image"], axcodes="SPL"),
                TemporalRandomCrop(keys=["image"], size=16),
                transforms.SpatialPadd(keys=["image"], spatial_size=(16, 256, 256), method="symmetric",
                                       mode="constant"),
                # 中心裁剪
                transforms.CenterSpatialCropd(keys=["image"], roi_size=(16, 256, 256)),
                transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255, b_min=-1.0, b_max=1.0),
                transforms.ToTensor(),
            ])
            return TrainDataset(data_path=os.path.join(args.data_path, phase),
                               transform=mpect_transforms, use_cond=args.stru_cond)
        else:
            mpect_transforms = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image"], allow_missing_keys=True),
                    transforms.EnsureChannelFirstD(keys=['image']),
                    transforms.Spacingd(keys=["image"], pixdim=(1.5, 1.5, 5.0)),
                    transforms.Orientationd(keys=["image"], axcodes="SPL"),
                    transforms.SpatialPadd(keys=["image"], spatial_size=(16, 256, 256), method="symmetric",
                                           mode="constant"),
                    transforms.CenterSpatialCropd(keys=["image"], roi_size=(16, 256, 256)),
                    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255, b_min=-1.0, b_max=1.0),
                    transforms.ToTensor(),
                ]
            )
            return TestDataset(data_path=os.path.join(args.data_path, phase),
                              transform=mpect_transforms, use_cond=args.stru_cond)
