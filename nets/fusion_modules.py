import torch
import torch.nn as nn


class DAAF(nn.Module):
    """Public DAAF fusion interface.

    This public version preserves the training and inference interface used by
    the detector. The detailed research implementation will be released after
    the manuscript review process is completed.
    """

    def __init__(self, channels, layer_idx=0, reduction=16):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_cnn, feat_transformer):
        fused = self.fusion(torch.cat([feat_cnn, feat_transformer], dim=1))
        weight = self.context(fused)
        return fused * weight + feat_cnn * (1.0 - weight)


def build_fusion(fusion_type, channels, layer_idx=0, reduction=16):
    fusion_type = fusion_type.lower()
    if fusion_type not in {"daaf", "default"}:
        raise ValueError("Only the public DAAF fusion interface is available in this release.")
    return DAAF(channels=channels, layer_idx=layer_idx, reduction=reduction)
