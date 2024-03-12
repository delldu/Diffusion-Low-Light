import torch
import torch.nn as nn

# haar wavelet
class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat((x_LL, x_HL, x_LH, x_HH), dim=0)

class IDWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, x):
        B1, C1, H1, W1 = x.size()
        B2, C2, H2, W2 = B1 // 4, C1, 2 * H1, 2 * W1
        x1 = x[0:B2, :, :, :] / 2.0
        x2 = x[B2 : B2 * 2, :, :, :] / 2.0
        x3 = x[B2 * 2 : B2 * 3, :, :, :] / 2.0
        x4 = x[B2 * 3 : B2 * 4, :, :, :] / 2.0

        h = torch.zeros([B2, C2, H2, W2]).float().to(x.device)

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h
