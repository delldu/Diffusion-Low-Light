import os
import numpy as np
import torch
import torch.nn as nn
from .mods import HFRM
from .wavelet import DWT, IDWT
from .unet import DiffusionUNet
from torch.nn import functional as F

import todos
import pdb


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class LowLightEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = DWT()
        self.high_enhance0 = HFRM(in_channels=3, out_channels=64)
        self.high_enhance1 = HFRM(in_channels=3, out_channels=64)

        self.load_weights()

    def forward(self, x):
        B, C, H, W = x.size()
        x = data_transform(x)
        # tensor [x] size: [1, 3, 512, 512], min: -1.0, max: -0.090196, mean: -0.800914

        x_dwt = self.dwt(x)
        # tensor [x_dwt] size: [4, 3, 256, 256], min: -2.0, max: 0.498039, mean: -0.400283

        x_l, x_h = x_dwt[:B, ...], x_dwt[B:, ...]
        x_h = self.high_enhance0(x_h)

        x_l_dwt = self.dwt(x_l)
        x_l_l, x_l_h = x_l_dwt[:B, ...], x_l_dwt[B:, ...]
        x_l_h = self.high_enhance1(x_l_h)
        x_h_dwt = self.dwt(x_h)

        # x_l_l.size() -- [1, 3, 128, 128]
        # x_l_h.size() -- [3, 3, 128, 128]
        # x_h.size() -- [3, 3, 256, 256]
        # x_h_dwt.size() -- [12, 3, 128, 128]

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # image_lowlight_encoder.onnx(image) ==> list[19, 3, H4, W4]
        # image_lowlight_denoise.onnx(LL_LL) ==> [1, 3, H4, W4]
        # image_lowlight_decoder.onnx([1, 3, H4, H4], [3, 3, H4, W4], [3, 3, H2, H2]) ==> (image)

        return torch.cat((x_l_l, x_l_h, x_h_dwt), dim = 0)


    def load_weights(self, model_path="models/image_lowlight.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        sd = torch.load(checkpoint)
        # remove unet weights from sd
        new_sd = {}
        for k, v in sd.items():
            if not k.startswith("Unet."):
                new_sd[k] = v
        self.load_state_dict(new_sd)


class LowLightDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.idwt = IDWT()

    def forward(self, x_l_l, x_l_h, x_h_dwt):
        # x_l_l.size() -- [1, 3, 128, 128]
        # x_l_h.size() -- [3, 3, 128, 128]
        # x_h_dwt.size() [12, 3, 128, 128]
        x_h = self.idwt(x_h_dwt)
        x_l = self.idwt(torch.cat((x_l_l, x_l_h), dim=0))  # size() -- [1, 3, 256, 256]
        x = self.idwt(torch.cat((x_l, x_h), dim=0))
        x = inverse_data_transform(x)  # size() -- [1, 3, 512, 512]
        return x


class DiffLLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 32
        # GPU memory 2G, 430ms

        self.encoder = LowLightEncoder()
        self.Unet = DiffusionUNet()
        self.decoder = LowLightDecoder()

        self.num_timesteps = 200
        self.sampling_timesteps = 10

        # get_beta_schedule
        betas = np.linspace(0.0001, 0.02, self.num_timesteps, dtype=np.float64)
        betas = torch.from_numpy(betas).float()
        # self.register_buffer("betas", torch.tensor(betas))
        self.register_buffer("betas", betas)
        # len(self.betas) -- 200

        self.register_buffer("one_betas_cumprod1", (1.0 - betas).cumprod(dim=0))
        betas = torch.cat([torch.zeros(1), betas], dim=0) # add zero at first
        self.register_buffer("one_betas_cumprod2", (1.0 - betas).cumprod(dim=0))

        skip = self.num_timesteps // self.sampling_timesteps  # 200//10 ==> 20
        self.reversed_seq = [i for i in range(0, self.num_timesteps, skip)][::-1] # reverse [0, 20, ..., 160, 180]
        self.reversed_next_seq = self.reversed_seq[1:] + [-1] # reverse [-1, 0, 20, ..., 140, 160]

    def resize_pad_tensor(self, x):
        # Need Resize ?
        B, C, H, W = x.size()
        s = min(min(self.MAX_H / H, self.MAX_W / W), 1.0)
        SH, SW = int(s * H), int(s * W)
        resize_x = F.interpolate(x, size=(SH, SW), mode="bilinear", align_corners=False)

        # Need Pad ?
        r_pad = (self.MAX_TIMES - (SW % self.MAX_TIMES)) % self.MAX_TIMES
        b_pad = (self.MAX_TIMES - (SH % self.MAX_TIMES)) % self.MAX_TIMES
        resize_pad_x = F.pad(resize_x, (0, r_pad, 0, b_pad), mode="replicate")
        return resize_pad_x

    def forward(self, x):
        # tensor [x] size: [1, 3, 416, 608], min: 0.0, max: 0.2, mean: 0.057457
        B2, C2, H2, W2 = x.size()
        x = self.resize_pad_tensor(x)

        x = self.encoder(x) # size() -- [16, 3, 104, 152]

        x_l_l = x[0:1, :, :, :] # size() -- [1, 3, 104, 152]
        x_l_h = x[1:4, :, :, :] # size() -- [3, 3, 104, 152]
        x_h_dwt = x[4:17, :, :]

        #################################################################################
        denoise_l_l = self.sample_training(x_l_l)  # size() -- [1, 3, 104, 152]
        #################################################################################

        pred_x = self.decoder(denoise_l_l, x_l_h, x_h_dwt)

        return pred_x[:, :, 0:H2, 0:W2].clamp(0.0, 1.0)

    def compute_alpha(self, t):
        a = self.one_betas_cumprod2.index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, eta: float=0.0):
        # tensor [x_cond] size: [1, 3, 104, 152], min: -4.0, max: -0.780392, mean: -3.203657

        B, C, H, W = x_cond.shape
        x = torch.randn(B, C, H, W, device=x_cond.device)
        xs = [x]

        # self.reversed_seq -- [180, 160, 140, 120, 100, 80, 60, 40, 20, 0]
        # self.reversed_next_seq -- [160, 140, 120, 100, 80, 60, 40, 20, 0, -1]
        for i, j in zip(self.reversed_seq, self.reversed_next_seq):
            t = (torch.ones(B) * i).to(x.device)
            next_t = (torch.ones(B) * j).to(x.device)
            at = self.compute_alpha(t.long())
            at_next = self.compute_alpha(next_t.long())
            xt = xs[-1].to(x.device)

            ####################################################################
            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)  # DiffusionUNet
            ####################################################################

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            # ==> c1 == 0.0
            c2 = ((1 - at_next) - c1**2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        # xs is list: len = 11
        #     tensor [item] size: [1, 3, 104, 152], min: -4.101145, max: 4.415229, mean: -0.004912
        #     tensor [item] size: [1, 3, 104, 152], min: -3.668343, max: 3.650547, mean: -0.018262
        #     tensor [item] size: [1, 3, 104, 152], min: -2.933483, max: 3.590278, mean: -0.12536
        #     tensor [item] size: [1, 3, 104, 152], min: -3.006614, max: 3.327058, mean: -0.408964
        #     tensor [item] size: [1, 3, 104, 152], min: -3.164008, max: 3.564155, mean: -0.441434
        #     tensor [item] size: [1, 3, 104, 152], min: -3.300746, max: 4.143057, mean: -0.526234
        #     tensor [item] size: [1, 3, 104, 152], min: -3.352117, max: 4.671752, mean: -0.517515
        #     tensor [item] size: [1, 3, 104, 152], min: -3.444924, max: 4.711062, mean: -0.473301
        #     tensor [item] size: [1, 3, 104, 152], min: -3.509325, max: 4.81771, mean: -0.486404
        #     tensor [item] size: [1, 3, 104, 152], min: -3.591407, max: 4.834415, mean: -0.489853
        #     tensor [item] size: [1, 3, 104, 152], min: -3.591892, max: 4.836276, mean: -0.489913
        return xs[-1]
