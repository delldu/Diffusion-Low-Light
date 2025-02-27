import torch
import torch.nn as nn
import warnings
import math

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention " "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = DepthConv(in_ch=dim, out_ch=dim)
        self.key = DepthConv(in_ch=dim, out_ch=dim)
        self.value = DepthConv(in_ch=dim, out_ch=dim)

        # self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states, ctx):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)

        query_layer = mixed_query_layer.permute(0, 2, 1, 3)
        key_layer = mixed_key_layer.permute(0, 2, 1, 3)
        value_layer = mixed_value_layer.permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = self.softmax(attention_scores)

        # attention_probs = self.dropout(attention_probs)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        return ctx_layer


class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch, out_channels=in_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 1), stride=(1, 1), padding=0, groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class Dilated_Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        sequence = list()
        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, dilation=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=3, dilation=(3, 3)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, dilation=(1, 1)),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x) + x

        return out


class HFRM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_head = DepthConv(in_channels, out_channels)

        self.dilated_block_LH = Dilated_Resblock(out_channels, out_channels)
        self.dilated_block_HL = Dilated_Resblock(out_channels, out_channels)

        self.cross_attention0 = CrossAttention(out_channels, num_heads=8)
        self.dilated_block_HH = Dilated_Resblock(out_channels, out_channels)
        self.conv_HH = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)
        self.cross_attention1 = CrossAttention(out_channels, num_heads=8)

        self.conv_tail = DepthConv(out_channels, in_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        
        residual = x

        x = self.conv_head(x)
        x_HL, x_LH, x_HH = x[: b // 3, ...], x[b // 3 : 2 * b // 3, ...], x[2 * b // 3 :, ...]

        x_HH_LH = self.cross_attention0(x_LH, x_HH)
        x_HH_HL = self.cross_attention1(x_HL, x_HH)

        x_HL = self.dilated_block_HL(x_HL)
        x_LH = self.dilated_block_LH(x_LH)

        x_HH = self.dilated_block_HH(self.conv_HH(torch.cat((x_HH_LH, x_HH_HL), dim=1)))

        out = self.conv_tail(torch.cat((x_HL, x_LH, x_HH), dim=0))

        return out + residual
