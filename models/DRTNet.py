import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


# ===================== 基础工具组件 =====================
class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


# 可学习位置编码 (论文：嵌入特征图，丰富表征)
class LearnablePosEmb(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, dim, h, w))

    def forward(self, x):
        return x + self.pos_emb


# 多层感知机 MLP
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ===================== 核心1：矩形交叉注意力 RCA (论文图3.3b) =====================
# 垂直矩形交叉注意力 V-RCA
class VerticalRCA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        # QKV 线性投影 (HSI + MSI 交叉注意力)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, hsi_feat, msi_feat):
        B, C, H, W = hsi_feat.shape
        # 重塑为窗口序列 (垂直矩形窗口)
        hsi_feat = rearrange(hsi_feat, 'b c h w -> b (h w) c')
        msi_feat = rearrange(msi_feat, 'b c h w -> b (h w) c')

        # Q(MSI), K(HSI), V(HSI) 交叉注意力
        q = self.to_q(msi_feat)
        k = self.to_k(hsi_feat)
        v = self.to_v(hsi_feat)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.out(out).reshape(B, C, H, W)


# 水平矩形交叉注意力 H-RCA
class HorizontalRCA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, hsi_feat, msi_feat):
        B, C, H, W = hsi_feat.shape
        hsi_feat = rearrange(hsi_feat, 'b c h w -> b (h w) c')
        msi_feat = rearrange(msi_feat, 'b c h w -> b (h w) c')

        # Q(HSI), K(MSI), V(MSI) 交叉注意力
        q = self.to_q(hsi_feat)
        k = self.to_k(msi_feat)
        v = self.to_v(msi_feat)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.out(out).reshape(B, C, H, W)


# ===================== 矩形Transformer块 RTB (论文图3.3a) =====================
class RTB(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=256, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 双向 RCA 并行
        self.v_rca = VerticalRCA(dim, heads, dim_head, dropout)
        self.h_rca = HorizontalRCA(dim, heads, dim_head, dropout)
        self.mlp = MLP(dim, mlp_dim, dropout)

    def forward(self, hsi_feat, msi_feat):
        # 残差 + 双向 RCA 融合
        feat_v = self.v_rca(hsi_feat, msi_feat)
        feat_h = self.h_rca(hsi_feat, msi_feat)
        attn_out = feat_v + feat_h  #

        # 层归一化 + MLP + 残差
        attn_out = self.norm1(attn_out)
        mlp_out = self.mlp(rearrange(attn_out, 'b c h w -> b (h w) c'))
        mlp_out = rearrange(mlp_out, 'b (h w) c -> b c h w', h=attn_out.shape[2])
        out = self.norm2(attn_out + mlp_out)
        return out


# ===================== 核心2：DRT 双分支矩形Transformer模块 =====================
class DRT(nn.Module):
    def __init__(self, dim, h, w, depth=3, heads=8, dim_head=64, mlp_dim=256):
        super().__init__()
        self.pos_emb = LearnablePosEmb(dim, h, w)  # 可学习位置编码
        self.layers = nn.ModuleList([RTB(dim, heads, dim_head, mlp_dim) for _ in range(depth)])

    def forward(self, hsi_feat, msi_feat):
        # 位置编码注入
        hsi_feat = self.pos_emb(hsi_feat)
        msi_feat = self.pos_emb(msi_feat)

        # 多层 RTB 迭代
        for rtb in self.layers:
            feat = rtb(hsi_feat, msi_feat)
            hsi_feat = hsi_feat + feat  # 残差融合
            msi_feat = msi_feat + feat

        # 双分支输出融合
        return hsi_feat + msi_feat


# ===================== 核心3：SAFA 尺度自适应特征聚合模块 =====================
class SAFA(nn.Module):
    def __init__(self, dim, compress_ratio=4, groups=4):
        super().__init__()
        self.dim = dim
        self.groups = groups

        # 融合块：空间+通道 池化
        self.avg_pool_sp = nn.AdaptiveAvgPool2d(1)
        self.max_pool_sp = nn.AdaptiveMaxPool2d(1)
        self.avg_pool_ch = nn.AdaptiveAvgPool2d(1)
        self.max_pool_ch = nn.AdaptiveMaxPool2d(1)

        # 跨维度融合 + 通道压缩
        self.compress = nn.Sequential(
            nn.Conv2d(dim * 4, dim // compress_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(dim // compress_ratio, dim, 1)
        )

        # 选择块：分组卷积 + 自适应权重
        self.group_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=groups)
        self.weight_a = nn.Parameter(torch.ones(dim))
        self.weight_b = nn.Parameter(torch.ones(dim))
        self.final_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

    def forward(self, x):
        # 空间/通道 池化
        avg_sp = self.avg_pool_sp(x)
        max_sp = self.max_pool_sp(x)
        avg_ch = self.avg_pool_ch(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        max_ch = self.max_pool_ch(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        #跨维度融合
        fuse_feat = torch.cat([avg_sp, max_sp, avg_ch, max_ch], dim=1)
        fuse_feat = self.compress(fuse_feat)

        # Softmax 自适应权重
        weight = torch.sigmoid(fuse_feat)
        weight_a = self.weight_a.view(1, -1, 1, 1) * weight
        weight_b = self.weight_b.view(1, -1, 1, 1) * (1 - weight)

        # 分组卷积 + 特征选择
        group_feat = self.group_conv(x)
        out = x * weight_a + group_feat * weight_b

        # 1x1卷积调整通道
        return self.final_conv(out)


# ===================== 核心4：CESR 对比增强光谱恢复机制 + 对比损失 =====================
class CESR(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 1)

    # 欧氏距离
    def euclidean_dist(self, x, y):
        return torch.sqrt(torch.sum((x - y) ** 2, dim=-1))

    # 划分正负样本
    def sample_split(self, anchor, hsi_feat, msi_feat, threshold):
        dist_hsi = self.euclidean_dist(anchor, hsi_feat)
        pos_mask = dist_hsi < threshold
        neg_mask = dist_hsi >= threshold
        return pos_mask, neg_mask

    # 对比损失
    def contrast_loss(self, anchor, pos, neg, temperature=0.07):
        sim_pos = F.cosine_similarity(anchor, pos, dim=-1) / temperature
        sim_neg = F.cosine_similarity(anchor, neg, dim=-1) / temperature
        loss = -torch.log(torch.exp(sim_pos) / (torch.exp(sim_pos) + torch.exp(sim_neg).sum()))
        return loss.mean()

    def forward(self, fuse_feat, lr_hsi, hr_msi):
        anchor = self.proj(fuse_feat).flatten(2)  # 锚点：融合光谱特征
        hsi_feat = lr_hsi.flatten(2)
        msi_feat = hr_msi.flatten(2)

        # 自适应阈值
        dist = self.euclidean_dist(anchor, hsi_feat)
        threshold = dist.mean()

        # 正负样本
        pos_mask, neg_mask = self.sample_split(anchor, hsi_feat, msi_feat, threshold)
        loss = self.contrast_loss(anchor[:, pos_mask], hsi_feat[:, pos_mask], msi_feat[:, neg_mask])
        return fuse_feat, loss


# ===================== 论文损失函数 =====================
class DRTLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # MSE损失权重
        self.beta = beta  # 光谱重建损失权重
        self.gamma = gamma  # 对比损失权重
        self.mse = nn.MSELoss()
        self.spec_conv = nn.Conv2d(1, 1, 3, padding=1)  # 光谱重建卷积

    def forward(self, pred, gt, cesr_loss):
        # MSE损失
        loss_mse = self.mse(pred, gt)

        # 光谱重建损失
        pred_spec = self.spec_conv(pred)
        gt_spec = self.spec_conv(gt)
        loss_spec = self.mse(pred_spec, gt_spec)

        # 总损失
        total_loss = self.alpha * loss_mse + self.beta * loss_spec + self.gamma * cesr_loss
        return total_loss, loss_mse, loss_spec, cesr_loss


# ===================== 最终DRTnet 主网络 =====================
class DRTnet(nn.Module):
    def __init__(self,
                 scale_ratio=4,
                 n_bands=102,  # HSI波段数
                 msi_bands=4,  # MSI波段数
                 dim=64,  # 基础通道数
                 depth=3,  # DRT层数
                 heads=8):
        super().__init__()
        self.scale_ratio = scale_ratio

        # 1. 输入投影
        self.hsi_proj = nn.Conv2d(n_bands, dim, 3, padding=1)
        self.msi_proj = nn.Conv2d(msi_bands, dim, 3, padding=1)

        # 2. 核心DRT双分支矩形Transformer
        self.drt = DRT(dim=dim, h=192, w=192, depth=depth, heads=heads)

        # 3. SAFA尺度自适应特征聚合
        self.safa = SAFA(dim=dim)

        # 4. CESR对比增强光谱恢复
        self.cesr = CESR(dim=dim)

        # 5. 输出重建
        self.out_conv = nn.Conv2d(dim, n_bands, 3, padding=1)

    def forward(self, x_lr, x_hr):
        """
        x_lr: 低分辨率高光谱 LR-HSI [B, n_bands, H/4, W/4]
        x_hr: 高分辨率多光谱 HR-MSI [B, msi_bands, H, W]
        """
        # 上采样LR-HSI
        x_lr_up = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bicubic')

        # 特征投影
        hsi_feat = self.hsi_proj(x_lr_up)
        msi_feat = self.msi_proj(x_hr)

        # 1. DRT 双分支矩形Transformer
        fuse_feat = self.drt(hsi_feat, msi_feat)

        # 2. SAFA 尺度自适应聚合
        fuse_feat = self.safa(fuse_feat)

        # 3. CESR 对比增强 + 对比损失
        fuse_feat, cesr_loss = self.cesr(fuse_feat, x_lr_up, x_hr)

        # 最终重建
        pred_hsi = self.out_conv(fuse_feat)

        return pred_hsi, cesr_loss


# ===================== 构建函数 =====================
def make_drtnet():
    return DRTnet(
        scale_ratio=4,
        n_bands=102,
        msi_bands=4,
        dim=64,
        depth=3,
        heads=8
    )


if __name__ == '__main__':
    model = make_drtnet()
    print(model)
    # 测试输入
    lr_hsi = torch.randn(1, 102, 48, 48)
    hr_msi = torch.randn(1, 4, 192, 192)
    out, cesr_loss = model(lr_hsi, hr_msi)
    print(f"输出形状: {out.shape}")