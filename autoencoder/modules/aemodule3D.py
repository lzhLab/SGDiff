import math

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from autoencoder.ae_util import silu, Normalize, SiLU


# 残差网络模块
class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 conv_shortcut=False,
                 dropout=0.0,
                 norm_type='group',
                 padding_type='replicate'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type)
        self.conv2 = SamePadConv3d(out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x+h


class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3   # (3,3,3)
        if isinstance(stride, int):
            stride = (stride,) * 3             # (1,1,1)

        # 假设输入形状能被步幅整除
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())

        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)
        self.weight = self.conv.weight

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type
        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))


# 编码器部分
class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample, z_channels, double_z, image_channel=3, norm_type='group',
                 padding_type='replicate'):
        """
        :param n_hiddens:  模型隐藏层通道
        :param downsample: 下采样倍数 [4,8,8]
        :param z_channels: 潜在向量的通道
        :param double_z:   z通道翻倍
        :param image_channel: 图像通道
        :param norm_type: 标准化类型
        :param padding_type: 填充类型
        """
        super().__init__()
        # downsample:[4,8,8] -> n_times_downsample:[2,3,3]
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        # 卷积层
        self.conv_blocks = nn.ModuleList()
        # 3
        max_ds = n_times_downsample.max()
        # 初始卷积
        self.conv_first = SamePadConv3d(image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)

        for i in range(max_ds):       # i:0,1,2
            block = nn.Module()
            # 32,64,128
            in_channels = n_hiddens * 2 ** i
            # 64,128,256
            out_channels = n_hiddens * 2 ** (i + 1)
            # (2,2,2),(2,2,2),(1,2,2)
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            # 下采样  (b,in_c,d,h,w) -> (b,out_c,d/2,h/2,w/2)
            block.down = SamePadConv3d(in_channels, out_channels, 4, stride=stride, padding_type=padding_type)
            # 残差块  (b,out_c,d,h,w) -> (b,out_c,d,h,w)
            block.res = ResBlock(out_channels, out_channels, norm_type=norm_type)
            # 将下采样块和卷积块加入到卷积模块中
            self.conv_blocks.append(block)
            # [1,2,2] [0,1,1] [-1,0,0]
            n_times_downsample -= 1

        # out_c:256 -> 8
        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type),
            SiLU(),
            SamePadConv3d(out_channels, 2 * z_channels if double_z else z_channels,
                          kernel_size=3, stride=1, padding_type=padding_type)
        )
        self.out_channels = out_channels

    def forward(self, x):
        # x:[b,1,d,h,w] -> h:[b,n_hiddens,d,h,w]
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)   # 最终h:[1,4,16,64,64]
        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, z_channels, image_channel=3, norm_type='group'):
        """
        :param n_hiddens: 同encoder 32
        :param upsample:  同encoder [4,8,8]
        :param z_channels:  同encoder 4
        :param image_channel:  1
        :param norm_type:
        """
        super().__init__()
        # [2,3,3]
        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        # 3
        max_us = n_times_upsample.max()
        # 4
        in_channels = z_channels
        # 初始化卷积层
        self.conv_blocks = nn.ModuleList()
        for i in range(max_us):
            block = nn.Module()
            # 4,256,128
            in_channels = in_channels if i == 0 else n_hiddens * 2 ** (max_us - i + 1)
            # 256,128,64
            out_channels = n_hiddens * 2 ** (max_us - i)
            # (2,2,2),(2,2,2),(1,2,2)
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            block.up = SamePadConvTranspose3d(in_channels, out_channels, 4, stride=us)
            block.res1 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            block.res2 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.conv_out = SamePadConv3d(out_channels, image_channel, kernel_size=3)

    def forward(self, x):
        h = x    # x: [1,4,4,32,32]
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)  # 4 -> 256,4->8,32->64,32->64 上采样
            h = block.res1(h)  # 256->128,8->16,64->128,64->128
            h = block.res2(h)  # 128->64,16->16,128->256,128->256
        h = self.conv_out(h)   # h: 1,1,16,256,256
        return h


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)