import torch.nn as nn
import functools
from autoencoder.ae_util import ActNorm


# 判别网络
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=64, n_layers=3, use_actnorm=False, getIntermFeat=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- 输入图像通道数
            ndf (int)       -- 最后一个卷积层的滤波器数量ndf
            n_layers (int)  -- 卷积层的数量
            norm_layer      -- 是否使用ActNorm层进行归一化的标志use_actnorm
        """
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        # 根据use_actnorm的值，选择合适的归一化层,如果不使用ActNorm,则选择batchNorm2d作为归一化层
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        # 根据归一化层的类型，确定是否需要使用偏置参数
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        # 定义卷积核大小为4，填充为1
        kw = 4
        padw = 1
        # 创建一个sequence序列，用于存储鉴别器层
        # 并添加第一个卷积层核激活函数到序列中,该层将输入图像通道数转换为ndf
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]]
        # 用于记录上一层和当前层的滤波器数量的倍数
        nf_mult = 1
        nf_mult_prev = 1
        # 使用循环从1到n_layers-1,逐渐增加滤波器数量
        for n in range(1, n_layers):
            # 在循环中，更新nf_mult和nf_mult_prev的值，分别为当前层和上一层的滤波器数量的倍数
            # 其中当前层的滤波器数量的倍数不超过8。
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            # 在循环中，添加卷积层、归一化层和激活函数层到序列中
            sequence += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]]
        # 在循环结束后，更新nf_mult_prev和nf_mult的值，分别为上一层和最后一层的滤波器数量的倍数，
        # 其中最后一层的滤波器数量的倍数不超过8。
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        # 添加最后一层卷积层、归一化层和激活函数层到序列中
        sequence += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]]
        # 添加一个卷积层到序列中，将最后一层的滤波器数量转换为1，用于输出判别结果。
        sequence += [[nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            # 将序列转换为nn.Sequential类型，并赋值给类的属性self.main
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            """Standard forward."""
            return self.model(input)