import cv2
import torch
import random
import numpy as np
from einops import rearrange
from monai.config import KeysCollection
from typing import Mapping, Hashable, Dict
from monai.transforms.transform import MapTransform
from monai.config.type_definitions import NdarrayOrTensor


# 随机裁剪连续16帧
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


# 获取高频结构条件
def get_sobel(img):
    image = (img + 1) * 127.5
    image1 = rearrange(image, "c t h w -> t h w c").numpy()
    img_sobels = []
    for j in range(image1.shape[0]):
        # threshold_random = random.randint(1, 10)
        # bilateralFilter_random = random.randint(1, 30)
        # nplabs = cv2.bilateralFilter(image1[j], 10, bilateralFilter_random, bilateralFilter_random)
        # nplabs = np.uint8(nplabs)
        x = cv2.Sobel(image1[j], cv2.CV_16S, 1, 0,)
        y = cv2.Sobel(image1[j], cv2.CV_16S, 0, 1,)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        nplabs = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        # nplabs[nplabs < threshold_random] = 0
        # nplabs = (nplabs - nplabs.min() + 1e-12) / (nplabs.max() - nplabs.min() + 1e-8)
        img_sobels.append(nplabs)
    img_sobels = np.array(img_sobels)[:, :, :, None]
    image_sobels = torch.tensor(rearrange(img_sobels, "t h w c -> c t h w"))
    image_sobels = image_sobels / 127.5 - 1

    return image_sobels