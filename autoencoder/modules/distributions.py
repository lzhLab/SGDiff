import torch
import numpy as np


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        """
        :param parameters: 是现状为[batch_size, z_channels * 2 , z_d, z_h, z_w]的均值和方差对数
        :param deterministic:
        """
        self.parameters = parameters
        # 分割均值和对数方差
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        # 方差限制在-30.0 ~ 20.0之间
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        # 计算标准差
        self.std = torch.exp(0.5 * self.logvar)
        # 计算方差
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    # 用于采样
    def sample(self, noise=None):
        if noise is None:
            noise = torch.randn(self.mean.shape)

        x = self.mean + self.std * noise.to(device=self.parameters.device)
        return x

    # 计算KL散度
    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(torch.pow(self.mean - other.mean, 2) / other.var
                                       + self.var / other.var - 1.0 - self.logvar + other.logvar, dim=[1, 2, 3])

    # 计算负对数似然
    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


# 计算两个高斯分布之间的KL散度函数
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    计算两个高斯分布之间的KL散度。
    形状会自动广播, 因此可以将批次与标量进行比较，以及其他用例
    """
    tensor = None
    # 首先检查这四个参数中是否有一个是torch.Tensor类型的对象，
    # 如果有，则将其赋值给变量tensor，并跳出循环
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"


    # 将logvar1和logvar2强制转换为Tensor类型，以确保能够进行后续的计算。
    # 如果logvar1或logvar2是标量，则将其转换为与tensor相同的设备上的Tensor。
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    # 使用广播机制计算KL散度的各个部分，并返回计算结果。
    return 0.5 * (
            -1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )