import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from autoencoder.losses.lpips import LPIPS
from autoencoder.modules.discriminator import NLayerDiscriminator
from autoencoder.modules.aemodule3D import weights_init


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
            torch.mean(torch.nn.functional.softplus(-logits_real)) +
            torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self,
                 disc_start,
                 logvar_init=0.0,
                 kl_weight=1.0,
                 pixelloss_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=1,
                 disc_factor=1.0,
                 disc_weight=1.0,
                 perceptual_weight=1.0,  # 感知损失权重
                 use_actnorm=False,
                 disc_conditional=False,
                 disc_loss="hinge",
                 max_bs=None, ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.max_bs = max_bs

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, global_step, last_layer=None, cond=None,
                split="train", weights=None, ):
        """
        :param inputs:               输入原图像
        :param reconstructions:      重建图像
        :param posteriors:
        :param optimizer_idx:        优化器索引0:生成, 1:判别器
        :param global_step:
        :param last_layer:
        :param cond:
        :param split:
        :param weights:
        :return:
        """
        if inputs.dim() == 5:
            inputs = rearrange(inputs, 'b c t h w -> (b t) c h w')
        if reconstructions.dim() == 5:
            reconstructions = rearrange(reconstructions, 'b c t h w -> (b t) c h w')
        # l1重建损失
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        # 如果感知损失权重大于0
        if self.perceptual_weight > 0:
            if self.max_bs is not None and self.max_bs < inputs.shape[0]:
                input_list = torch.split(inputs, self.max_bs, dim=0)
                reconstruction_list = torch.split(reconstructions, self.max_bs, dim=0)
                p_losses = [self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous()) for
                            inputs, reconstructions in zip(input_list, reconstruction_list)]
                p_loss = torch.cat(p_losses, dim=0)
            else:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self,
                 disc_start,  # 判别器在训练步骤到达多少时，才有判别器损失
                 codebook_weight=1.0,  # codebook感知权重
                 pixelloss_weight=1.0,
                 disc_num_layers=3,  # 判别器当中卷积网络的层数
                 disc_in_channels=1,  # 判别器输入图像通道
                 disc_factor=1.0,
                 gan_feat_weight=0.0,
                 l1_weight=1.0,
                 disc_weight=1.0,  # 判别网络权重
                 perceptual_weight=1.0,  # 感知损失权重
                 use_actnorm=False,  # 是否使用激活归一化
                 disc_conditional=False,
                 disc_ndf=64,  # 判别器网络通道
                 disc_loss="hinge"
                 ):
        super(VQLPIPSWithDiscriminator, self).__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.gan_feat_weight = gan_feat_weight
        self.l1_weight = l1_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 ndf=disc_ndf,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 getIntermFeat=True).apply(weights_init)

        self.discriminator_iter_start = disc_start
        # 对于D来说，只有当D(x) < 1的正向样本，以及D(G(z)) > -1的负样本才会对结果产生影响
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        # vanilla: sigmoid + BCELoss
        # Ld = -[ylogD(x)+(1-y)log(1-D(G(Z)))]
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")

        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self,
                codebook_loss,  # codebook loss
                inputs,  # 输入图像
                reconstructions,  # 重建图像
                optimizer_idx,  # 0：生成器 1：判别器
                global_step,  # 全局步数
                last_layer=None,  # 解码器卷积权重
                cond=None,
                split="train"
                ):

        if inputs.dim() == 5:
            inputs = rearrange(inputs, 'b c t h w -> (b t) c h w')
        if reconstructions.dim() == 5:
            reconstructions = rearrange(reconstructions, 'b c t h w -> (b t) c h w')
        # 1.重建损失
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        # 2.感知损失
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_weight * self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + p_loss
        else:
            p_loss = torch.tensor([0.0])

        # 交叉熵损失
        nll_loss = torch.mean(rec_loss)

        # now the GAN part
        if optimizer_idx == 0:
            if cond is None:
                assert not self.disc_conditional
                logits_fake, pred_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake, pred_fake = self.video_discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            # GAN特征匹配损失 - 调整特征使得在鉴别器上获得相同的预测结果。
            gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)

            if self.discriminator_weight > 0:
                logits_real, pred_real = self.discriminator(inputs.contiguous())
                for i in range(len(pred_fake) - 1):
                    gan_feat_loss += feat_weights * F.l1_loss(pred_fake[i], pred_real[i].detach()) * \
                                     (self.discriminator_weight > 0)

            gan_feat_loss = disc_factor * self.gan_feat_weight * gan_feat_loss

            # 总损失 = 重建损失(包含感知损失) + 判别损失 + codebook损失 + 特征匹配损失
            loss = nll_loss + disc_factor * d_weight * g_loss + self.codebook_weight * codebook_loss.mean() \
                   + gan_feat_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/gen_feat_loss".format(split): gan_feat_loss.detach().mean(),
                   "{}/perceptual_loss".format(split): p_loss.detach().mean()
                   }

            return loss, log

        if optimizer_idx == 1:
            # 训练判别器
            if cond is None:
                logits_real, _ = self.discriminator(inputs.contiguous().detach())
                logits_fake, _ = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real, _ = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake, _ = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean(),
                   }
            return d_loss, log