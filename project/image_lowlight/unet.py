import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import pdb

# This script is from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


class DictToClass(object):
    def __init__(self, _obj):
        if _obj:
            self.__dict__.update(_obj)


def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0.0)
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:  # Support torch.jit.script
            self.nin_shortcut = nn.Identity()

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class DiffusionUNet(nn.Module):
    """
    LOLv1.yml

    model:
        in_channels: 3
        out_ch: 3
        ch: 64
        ch_mult: [1, 2, 3, 4]
        num_res_blocks: 2
        dropout: 0.0
        ema_rate: 0.999
        ema: True
        resamp_with_conv: True
    """
    def __init__(self):
        super().__init__()
        config = DictToClass(
            {
                "in_channels": 3,
                "ch": 64,
                "out_ch": 3,
                "ch_mult": [1, 2, 3, 4],
                "num_res_blocks": 2,
                "dropout": 0.0,
                "ema_rate": 0.999,
                "ema": True,
                "resamp_with_conv": True,
            }
        )

        out_ch, ch_mult = config.out_ch, tuple(config.ch_mult)
        dropout = config.dropout
        in_channels = config.in_channels * 2

        self.ch = config.ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)  # (1, 2, 3, 4)
        self.num_res_blocks = config.num_res_blocks  # 2
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList(
            [
                nn.Linear(self.ch, self.temb_ch),
                nn.Linear(self.temb_ch, self.temb_ch),
            ]
        )

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions): # 4
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * ch_mult[i_level]
            for j_block in range(self.num_res_blocks): # 2
                block.append(
                    ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout)
                )
                block_in = block_out
                if i_level == 2:
                    attn.append(AttnBlock(block_in))
                else:
                    attn.append(nn.Identity())

            # save down to self.down
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            else:
                down.downsample = nn.Identity()
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)): # 4
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.ch * ch_mult[i_level]
            skip_in = self.ch * ch_mult[i_level]
            for j_block in range(self.num_res_blocks + 1): # 3
                if j_block == self.num_res_blocks:
                    skip_in = self.ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if i_level == 2:
                    attn.append(AttnBlock(block_in))
                else:
                    attn.append(nn.Identity())

            # save up to self.up
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
            else:
                up.upsample = nn.Identity()
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs: List[torch.Tensor] = [self.conv_in(x)]
        # for i_level in range(self.num_resolutions): # 4
        #     layer = self.down[i_level]

        #     for j_block in range(self.num_res_blocks): # 2
        #         h = layer.block[j_block](hs[-1], temb)
        #         if i_level == 2:
        #             h = layer.attn[j_block](h)
        #         hs.append(h)
        #     if i_level != 3:
        #         hs.append(layer.downsample(hs[-1]))

        for i_level, layer in enumerate(self.down): # 4
            # j_block == 0
            h = layer.block[0](hs[-1], temb)
            if i_level == 2:
                h = layer.attn[0](h)
            hs.append(h)

            # j_block == 1
            h = layer.block[1](hs[-1], temb)
            if i_level == 2:
                h = layer.attn[1](h)
            hs.append(h)

            if i_level != 3:
                hs.append(layer.downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        # for i_level in reversed(range(self.num_resolutions)): # [3, 2, 1, 0]
        #     up_m = self.up[i_level]

        #     for j_block in range(self.num_res_blocks + 1):
        #         t = torch.cat([h, hs.pop()], dim=1) # temp
        #         h = up_m.block[j_block](t, temb)
        #         if i_level == 2:
        #             h = up_m.attn[j_block](h)
        #     if i_level != 0:
        #         h = up_m.upsample(h)
        for i in range(self.num_resolutions):
            h = self.up_layer(self.num_resolutions - i - 1, h, temb, hs) # layer_i

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def up_layer(self, i: int, h, temb, hs: List[torch.Tensor]):
        """ugly for torch.jit.script no reversed(), oh oh oh !!!"""
        for i_level, layer in enumerate(self.up):
            if i_level == i:
                # j_block == 0
                t = torch.cat([h, hs.pop()], dim=1) # temp
                h = layer.block[0](t, temb)
                if i_level == 2:
                    h = layer.attn[0](h)

                # j_block == 1
                t = torch.cat([h, hs.pop()], dim=1) # temp
                h = layer.block[1](t, temb)
                if i_level == 2:
                    h = layer.attn[1](h)

                # j_block == 2
                t = torch.cat([h, hs.pop()], dim=1) # temp
                h = layer.block[2](t, temb)
                if i_level == 2:
                    h = layer.attn[2](h)

                if i_level != 0:
                    h = layer.upsample(h)

        return h