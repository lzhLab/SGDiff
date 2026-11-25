import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import instantiate_from_config
from .modules.distributions import DiagonalGaussianDistribution
from autoencoder.modules.aemodule3D import Encoder, Decoder, SamePadConv3d


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,    # 量化嵌入空间维数
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 monitor=None, std=1., mean=0., prob=0.2, *args, **kwargs):
        super(AutoencoderKL, self).__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig.encoder.params)
        self.decoder = Decoder(**ddconfig.decoder.params)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        # 将编码器隐空间维度(z_channels)映射到embedding维度(embed_dim)
        self.quant_conv = SamePadConv3d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        # 将embedding维度(embed_dim)映射回解码器隐空间维度(z_channels)
        self.post_quant_conv = SamePadConv3d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.std = std
        self.mean = mean
        self.prob = prob
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        try:
            self._cur_epoch = sd['epoch']
            sd = sd['state_dict']
        except:
            pass
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x, **kwargs):
        # x:[b,1,d,h,w] -> h:[b,8,d/4,h/8,w/8]
        h = self.encoder(x)
        # h:[b,8,d/4,h/8,w/8] -> moments:[b,2*embed_dim,d/4,h/8,w/8]
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, **kwargs):
        # z:[b,emb_dim,z_d,z_h,z_w]
        z = self.post_quant_conv(z)
        dec = self.decoder(z)   # h:[1,1,16,256,256]
        return dec

    def forward(self, input, sample_posterior=True, *args, **kwargs):
        # encode编码得到服从该均值方差的高斯分布
        posterior = self.encode(input)
        # 从服从该均值方差的高斯分布中采样
        if sample_posterior:
            z = posterior.sample()  # z:[1,4,4,32,32]
        else:
            z = posterior.mode()
        # 解码输出图像
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 4:
            x = x[..., None]
        # 将张量x转换为连续内存格式，并将其类型转换为浮点型
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dicr_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dicr_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            # log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x