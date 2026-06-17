import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class baseline(nn.Module):
    """Plain DRT baseline without rectangular transformer, multiresolution paths, or contrastive learning.

    This model keeps the DRT input and output contract:
    input:  LR-HSI x_lr and HR-MSI x_hr;
    output: reconstructed HR-HSI and a second tensor for compatibility with
    existing training and validation code.
    """

    uses_contrastive_learning = False
    uses_rectangular_transformer = False
    uses_multiresolution_features = False

    def __init__(
        self,
        arch="baseline",
        scale_ratio=4,
        n_select_bands=5,
        n_bands=103,
        dataset=None,
        n_colors=None,
        channels=64,
        num_blocks=8,
    ):
        super().__init__()
        self.arch = arch
        self.scale_ratio = scale_ratio
        self.n_select_bands = n_select_bands
        self.n_bands = n_bands
        self.dataset = dataset

        self.lr_head = nn.Sequential(
            nn.Conv2d(n_bands, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(channels),
        )
        self.hr_head = nn.Sequential(
            nn.Conv2d(n_select_bands, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(channels),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            *[ResidualBlock(channels) for _ in range(num_blocks)],
        )
        self.reconstruction = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, n_bands, kernel_size=3, padding=1),
        )
        self.spectral_refine = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, padding=1),
        )

    def forward(self, x_lr, x_hr):
        target_size = x_hr.shape[-2:]
        x_lr_up = F.interpolate(
            x_lr,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        lr_feat = self.lr_head(x_lr_up)
        hr_feat = self.hr_head(x_hr)
        fused = self.fusion(torch.cat((lr_feat, hr_feat), dim=1))

        x = x_lr_up + self.reconstruction(fused)
        x = x + self.spectral_refine(x)
        return x, x


Baseline = baseline
DRTBaseline = baseline
