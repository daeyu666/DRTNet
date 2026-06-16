from __future__ import annotations

import torch
import torch.nn as nn

from models.IntmdSequential import IntermediateSequential
from models.transformer_rectangle import (
    FeedForward,
    LearnedPositionalEncoding,
    PreNorm,
    PreNormDrop,
    Residual,
)


class SquareWindowSelfAttention(nn.Module):
    """Two-branch self-attention with ordinary square local windows.

    This is used as the non-rectangle ablation for DRTNet.  The original
    rectangle transformer keeps separate horizontal/vertical rectangle masks;
    here every head uses the same square-window mask with a comparable area.
    """

    def __init__(
        self,
        dim,
        map_size,
        heads=8,
        qkv_bias=False,
        qk_scale=None,
        dropout_rate=0.0,
        window_size=12,
    ):
        super().__init__()
        self.num_heads = heads
        self.map_size = int(map_size)
        self.window_size = max(1, int(window_size))
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.out = {}

    def _square_window_mask(self, device):
        map_size = self.map_size
        n_tokens = map_size * map_size
        window_size = min(self.window_size, map_size)

        if window_size >= map_size:
            return torch.ones((n_tokens, n_tokens), dtype=torch.bool, device=device)

        mask = torch.zeros((n_tokens, n_tokens), dtype=torch.bool, device=device)
        for row in range(0, map_size, window_size):
            for col in range(0, map_size, window_size):
                row_end = min(row + window_size, map_size)
                col_end = min(col + window_size, map_size)
                rows = torch.arange(row, row_end, device=device)
                cols = torch.arange(col, col_end, device=device)
                idx = (rows[:, None] * map_size + cols[None, :]).reshape(-1)
                mask[idx[:, None], idx[None, :]] = True
        return mask

    def _attention_branch(self, tokens):
        residual = tokens
        bsz, n_tokens, channels = tokens.shape
        expected_tokens = self.map_size * self.map_size
        if n_tokens != expected_tokens:
            raise ValueError(
                "SquareWindowSelfAttention expected {} tokens for map_size={}, got {}".format(
                    expected_tokens, self.map_size, n_tokens
                )
            )

        qkv = (
            self.qkv(tokens)
            .reshape(bsz, n_tokens, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = self._square_window_mask(tokens.device).view(1, 1, n_tokens, n_tokens)
        attn = attn.masked_fill(~mask, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        tokens = (attn @ v).transpose(1, 2).reshape(bsz, n_tokens, channels)
        tokens = self.proj(tokens)
        tokens = self.proj_drop(tokens)
        return tokens + residual

    def forward(self, input):
        self.out["x"] = self._attention_branch(input["x"])
        self.out["y"] = self._attention_branch(input["y"])
        return self.out


class TransformerModel_square_window(nn.Module):
    def __init__(
        self,
        map_size,
        M_channel,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        window_size=12,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SquareWindowSelfAttention(
                                dim,
                                map_size=map_size,
                                heads=heads,
                                dropout_rate=attn_dropout_rate,
                                window_size=window_size,
                            ),
                        )
                    ),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))),
                ]
            )
        self.net = IntermediateSequential(*layers)
        self.input = {}
        self.output = {}
        self.map_size = int(map_size)
        self.linear_encoding = nn.Linear(M_channel, dim)
        self.linear_encoding_de = nn.Linear(dim, M_channel)
        self.position_encoding = LearnedPositionalEncoding(M_channel, dim, self.map_size * self.map_size)

    def forward(self, x, y):
        x_ = x.permute(0, 2, 3, 1).contiguous()
        y_ = y.permute(0, 2, 3, 1).contiguous()
        x_tokens = x_.view(x_.size(0), x_.size(2) * x_.size(1), -1)
        y_tokens = y_.view(y_.size(0), y_.size(2) * y_.size(1), -1)

        self.input["x"] = self.position_encoding(self.linear_encoding(x_tokens))
        self.input["y"] = self.position_encoding(self.linear_encoding(y_tokens))
        results = self.net(self.input)

        x_out = self.linear_encoding_de(results["x"]).permute(0, 2, 1).contiguous()
        y_out = self.linear_encoding_de(results["y"]).permute(0, 2, 1).contiguous()
        self.output["x"] = x_out.view(x_out.size(0), x_out.size(1), self.map_size, self.map_size)
        self.output["y"] = y_out.view(y_out.size(0), y_out.size(1), self.map_size, self.map_size)
        self.output["z"] = self.output["x"] + self.output["y"]
        return self.output
