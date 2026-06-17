import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_square_window import TransformerModel_square_window


class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class transformer_baseline(nn.Module):
    """Independent square-window Transformer baseline.

    Design choices:
    - square-window Transformer instead of rectangular Transformer;
    - a single working resolution, aligned to LR-HSI resolution;
    - no multiresolution DRT branch, no SAFA/multiscale pooling;
    - no contrastive learning branch.
    """

    uses_contrastive_learning = False
    uses_rectangular_transformer = False
    uses_square_window_transformer = True
    uses_multiresolution_features = False

    def __init__(
        self,
        arch="transformer_baseline",
        scale_ratio=4,
        n_select_bands=5,
        n_bands=103,
        dataset=None,
        n_colors=None,
        lr_map_size=32,
        channels=64,
        transformer_dim=64,
        transformer_depth=5,
        transformer_heads=8,
        transformer_window=12,
        num_refine_blocks=4,
    ):
        super().__init__()
        self.arch = arch
        self.scale_ratio = scale_ratio
        self.n_select_bands = n_select_bands
        self.n_bands = n_bands
        self.dataset = dataset
        self.lr_map_size = lr_map_size

        self.hsi_stem = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.msi_to_lr = nn.Sequential(
            nn.Conv2d(n_select_bands, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualConvBlock(channels),
            nn.Conv2d(channels, n_bands, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.transformer = TransformerModel_square_window(
            map_size=lr_map_size,
            M_channel=n_bands,
            dim=transformer_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            mlp_dim=n_bands,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
            window_size=transformer_window,
        )

        self.lr_fusion = nn.Sequential(
            nn.Conv2d(n_bands * 3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            *[ResidualConvBlock(channels) for _ in range(num_refine_blocks)],
            nn.Conv2d(channels, n_bands, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.hr_reconstruction = nn.Sequential(
            nn.Conv2d(n_bands * 2 + n_select_bands, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            *[ResidualConvBlock(channels) for _ in range(num_refine_blocks)],
            nn.Conv2d(channels, n_bands, kernel_size=3, padding=1),
        )

        self.spectral_refine = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, padding=1),
        )

    def _resize_lr_features(self, x):
        if x.shape[-2:] == (self.lr_map_size, self.lr_map_size):
            return x
        return F.interpolate(x, size=(self.lr_map_size, self.lr_map_size), mode="bilinear", align_corners=False)

    def forward(self, x_lr, x_hr):
        target_size = x_hr.shape[-2:]
        hsi_lr = self._resize_lr_features(self.hsi_stem(x_lr))
        msi_lr = self._resize_lr_features(self.msi_to_lr(x_hr))

        trans = self.transformer(hsi_lr, msi_lr)["z"]
        fused_lr = self.lr_fusion(torch.cat((hsi_lr, msi_lr, trans), dim=1))

        fused_hr = F.interpolate(fused_lr, size=target_size, mode="bilinear", align_corners=False)
        x_lr_up = F.interpolate(x_lr, size=target_size, mode="bilinear", align_corners=False)
        x = x_lr_up + self.hr_reconstruction(torch.cat((fused_hr, x_lr_up, x_hr), dim=1))
        x_spec = x + self.spectral_refine(x)
        return x_spec, x_spec


TransformerBaseline = transformer_baseline
