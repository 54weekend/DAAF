import torch
import torch.nn as nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = ConvBlock(channel, channel, kernel_size)
        self.conv2 = nn.Conv2d(channel, dim, 1, bias=False)

        self.transformer = nn.ModuleList([
            TransformerBlock(dim*patch_size[0]*patch_size[1], mlp_dim, dropout) for _ in range(depth)
        ])
        self.conv3 = ConvBlock(dim, channel)
        self.conv4 = ConvBlock(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        x = self.conv1(x)
        x = self.conv2(x)

        _, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, (h // self.ph) * (w // self.pw), self.ph * self.pw * x.shape[-1])

        for transformer in self.transformer:
            x = transformer(x)
        x = x.reshape(-1, h, w, c)

        x = x.permute(0, 3, 1, 2)

        x = self.conv3(x)
        x = torch.cat((x, y), dim=1)
        x = self.conv4(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=8, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attn(x, x, x)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class MobileViT(nn.Module):
    def __init__(self,  dims, channels,  expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()

        ph, pw = patch_size

        self.stem = ConvBlock(3, channels[0], stride=2)
        self.dark2 = ConvBlock(channels[0], channels[1], stride=2)

        self.dark3 = nn.Sequential(
            ConvBlock(channels[1], channels[2], stride=2),
            MobileViTBlock(dims[0], depth=2, channel=channels[2],
                           kernel_size=kernel_size, patch_size=patch_size,
                           mlp_dim=int(dims[0] * expansion))
        )

        self.dark4 = nn.Sequential(
            ConvBlock(channels[2], channels[3], stride=2),
            MobileViTBlock(dims[1], depth=4, channel=channels[3],
                           kernel_size=kernel_size, patch_size=patch_size,
                           mlp_dim=int(dims[1] * expansion))
        )

        self.dark5 = nn.Sequential(
            ConvBlock(channels[3], channels[4], stride=2),
            MobileViTBlock(dims[2], depth=3, channel=channels[4],
                           kernel_size=kernel_size, patch_size=patch_size,
                           mlp_dim=int(dims[2] * expansion))
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        feat1 = x
        x = self.dark4(x)
        feat2 = x
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3

def mobilevit_s():
    return MobileViT(
        dims=[144, 192, 240],
        channels=[32, 64, 128, 256,256]
    )
