import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic_blocks import *
from einops import rearrange
import numpy as np
import cv2
import math
from models.transformer_rectangle import TransformerModel_rectangle
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.MSA import MSAB #MSAB_TSTfirst
from models.SK import SKConv
from models.EMA import ema
from models.CGlayers import CG_Conv2d
from models.Multiscalepool import MultiScaleFusion_Pooling
pool_sizes = [1, 2, 4, 8]
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        self.relu = nn.ReLU()
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self, input):
        out =self.relu(self.depth_conv(input))  # 0523添加relu
        out = self.relu(self.point_conv(out))
        return out
class ConvMoudle1(nn.Module):
    def __init__(self, in_channels, u1_channel):
        super(ConvMoudle1, self).__init__()
        self.conv1 =  nn.Conv2d(in_channels, 512, 3, 1)
        self.relu = nn.ReLU()
        self.pos_emb = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(256, u1_channel, 3, 1, 1, bias=False),   # CG_Conv2d(256, 256, kernel_size = 3 ), #11.1
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(u1_channel, 256, 3, 1),  #11.1 CG_Conv2d(256, 256, kernel_size=3, padding=1),#
            nn.Conv2d(256,u1_channel,kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.pos_emb(x))  # x pos_emb torch.Size([1, 256, 4, 4])
        x = self.relu(self.conv4(x))  # x conv4 torch.Size([1, 308, 8, 8])
        return x
class ConvMoudle2_SK(nn.Module):
    expansion = 3  # outplanes = planes 204
    def __init__(self,outplanes,planes,n_bands,groups,stride=1,downsample=None):
        super(ConvMoudle2_SK,self).__init__()
        self.conv1=nn.Conv2d(planes, planes*self.expansion, 1,1, 0,bias=False)  #
        self.conv2 = SKConv(planes*self.expansion,planes,groups,stride)
        self.conv3 = nn.Conv2d(planes, outplanes , 1,1, 0,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dwconv = DepthWiseConv(planes, planes)
        self.conv4 = nn.Conv2d(n_bands,planes*self.expansion,1,1,0)
    def forward(self, x1,x2):
        shortcut =  self.dwconv(x2)  # x
        output = self.relu(self.conv4(x1))
        output2 = self.relu(self.conv1(x2)) #
        output = self.conv2(output,output2)
        output = self.conv3(output)
        output += shortcut
        return self.relu(output)
class ConvMoudle2(nn.Module):
    def __init__(self, in_channels, u2_channel):
        super(ConvMoudle2, self).__init__()
        self.conv1 = nn.Conv2d(1024, 512, 3, 1)
        self.conv2 = nn.Conv2d(512, 256, 3, 1)
        self.conv3 = nn.Conv2d(256, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, u2_channel, 3, 1)
        self.conv = nn.Conv2d(256, u2_channel, 1, 1)
        self.relu = nn.ReLU()
        self.pos_emb = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 2, bias=False),
            GELU(),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
        )
        self.dwconv = DepthWiseConv(1024, u2_channel)
    def forward(self, x):
        x1 = self.relu(self.conv1(x))  # 0523 conv1-->conv
        x1 = self.relu(self.pos_emb(x1))
        x1 = self.relu(self.conv(x1)) # 0523 conv4-->conv
        x2 = self.dwconv(x)
        x = x1 + x2
        return x
class ScaleStrip(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(ScaleStrip, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp
        self.deconv1 = nn.Conv2d(  # 条带卷积
            in_channels // 4, in_channels // 4, (1, 9), padding=(0, 4)  # in_channels // 8
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (1, 9), padding=(0, 4)
        )
        self.bn2 = BatchNorm(in_channels // 4 + in_channels // 4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()
        self._init_weight()
    def forward(self, x, inp=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.h_transform(x)
        x3 = self.deconv3(x3)
        x3 = self.inv_h_transform(x3)
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        '''x1 torch.Size([1, 77, 32, 32])
x2 torch.Size([1, 77, 32, 32])
x3 torch.Size([1, 77, 32, 32])
x4 torch.Size([1, 77, 32, 32])'''
        x = torch.cat((x1, x2, x3, x4), 1)
        return x
    def _init_weight(self):  # 初始化神经网络模型的权重参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 判断当前子模块是否是卷积层类型
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):  # 判断当前子模块是否是反卷积层类型
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):  # 判断当前子模块是否是 SynchronizedBatchNorm2d 类型（同步批归一化层）
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):  #
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))  # # 在特征图的右侧填充与宽度相同大小的0，实际上就是在水平方向上扩展了一倍
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]  # # 将特征图 reshape 成三维张量，最后一维是原宽度的两倍
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)  # 再将特征图 reshape 回原始形状，但是宽度变成了原来的两倍减一
        return x
    def inv_h_transform(self, x):  # 对h_transform后的张量进行恢复
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x
    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)
    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)
class MS_MSA(nn.Module):
    def __init__(
            self,
            dim, # n_bands*4
            dim_head, # 1024
            heads,  # 1
            dataset=None
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)  # dim_head  11.19
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim_head, dim_head, 3, 1, 1, bias=False, groups=dim_head),
            GELU(),
            nn.Conv2d(dim_head, dim, 3, 1, 1, bias=False, groups=1),   # dim_head
        )
        self.dim = dim
        self.ema = ema(dim)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
    def forward(self, x_in, dim):
        """
            x_in: [b,h,w,c]  # 0407输入为[b,c,h,w]，先做维度转换
            return out: [b,h,w,c]  # 输出仍为[b,c,h,w]
            """
        b, c, h, w = x_in.shape  # 0407 1,412,32,32
        x = x_in.permute(0, 2, 3, 1).reshape(b, h * w, c)  # Reshape to (b, h*w, c) 0407,,[1,1024,412]
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                      (q_inp, k_inp, v_inp))
        v = v
        q = q.transpose(-2, -1)  # q, torch.Size([1, 1, 1024, 1024])
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw  torch.Size([1, 1, 1024, 1024])
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)  # [1, 1024, 1024]
        out_c = self.proj(x).view(b, 32, 32, -1)  # view(b, h, w, c)  # [1, 8, 8, 618]  需要改成[1, 8, 8, 1024]？
        x_in = self.conv(x_in)
        x_in = x_in.permute(0, 2, 3, 1)
        out_p = self.pos_emb(v_inp.view(b, 32, 32, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p + x_in
        return out.permute(0, 3, 1, 2)
class MS_MSA2(nn.Module):
    def __init__(
            self,
            dim, # n_bands*4
            dim_head,
            heads,
            dataset=None
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim_head, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim_head, dim_head, 3, 1, 1, bias=False, groups=dim_head),
            GELU(),
            nn.Conv2d(dim_head, dim_head, 3, 1, 1, bias=False, groups=1),
        )
        self.dim = dim
        self.ema = ema(dim)
        self.conv = nn.Conv2d(dim, dim_head, 3, 1, 1, bias=False)
    def forward(self, x_in, dim):
        '''两个尺度'''
        """
            x_in: [b,h,w,c]  # 0407输入为[b,c,h,w]，先做维度转换
            return out: [b,h,w,c]  # 输出仍为[b,c,h,w]
        """
        b, c, h, w = x_in.shape  # 0407 1,412,32,32
        x = x_in.permute(0, 2, 3, 1).reshape(b, h * w, c)  # Reshape to (b, h*w, c) 0407,,[1,1024,412]
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                      (q_inp, k_inp, v_inp))
        v = v
        q = q.transpose(-2, -1)  # q, torch.Size([1, 1, 1024, 1024])
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw  torch.Size([1, 1, 1024, 1024])
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)  # [1, 1024, 1024]
        out_c = self.proj(x).view(b, h, h, -1)  # view(b, h, w, c)  # [1, 8, 8, 618]  需要改成[1, 8, 8, 1024]？
        x_in = self.ema(x_in)
        x_in = self.conv(x_in)
        x_in = x_in.permute(0, 2, 3, 1)
        out_p = self.pos_emb(v_inp.view(b, h, h, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p + x_in
        return out.permute(0, 3, 1, 2)
class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
class MCT_rectangle(nn.Module):
    def __init__(self,
                 arch,
                 scale_ratio,
                 n_select_bands,
                 n_bands,
                 dataset=None,
                  n_colors=None,
                 ):
        '''
    n_colors：总的通道数（或颜色通道数）。
    n_ovls：重叠的通道数，可能是用于某种共享特征或重用的通道数。
    n_subs：每组分配的通道数（子通道数）。
        '''
        """Load the pretrained ResNet and replace top fc layer."""
        super(MCT_rectangle, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands  # 103
        self.arch = arch
        self.n_select_bands = n_select_bands  # 5
        self.weight = nn.Parameter(torch.tensor([0.5]))
        self.conv_fus = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_spat = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_spec = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_select_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.D1 = nn.Sequential(  # 每次下采样4次
            nn.Conv2d(n_select_bands, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, n_bands, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.D2 = nn.Sequential(
            nn.Conv2d(n_bands, 156, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(156, 156, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(156, n_bands * 2, kernel_size=3, stride=2, padding=1), # stride=2
            nn.ReLU(),
            nn.Conv2d(n_bands * 2, n_bands * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        if dataset == 'paviaC':
            u1_channel = n_bands * 3 - 2
        elif dataset == 'PaviaU':
            u1_channel = n_bands * 3 - 1  # 308
        elif dataset == 'Botswana':
            u1_channel = n_bands * 3 - 3
        elif dataset == 'Urban':
            u1_channel = n_bands * 3 - 2  # 484
        elif dataset == 'DC':
            u1_channel = n_bands * 3 - 1
        elif dataset == 'IndianP':
            u1_channel = n_bands * 3 - 3
        elif dataset == 'KSC':
            u1_channel = n_bands * 3
        elif dataset == 'Chikusei':
            u1_channel = n_bands* 3 - 3
        self.U1 = nn.Sequential(
            nn.Conv2d(u1_channel, n_bands * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands * 2, n_bands * 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        if dataset == 'paviaC':
            self.u2_channel = n_bands * 2
        elif dataset == 'PaviaU':
            self.u2_channel = n_bands * 2 - 2  # 204
            self.conv3 = nn.Sequential(   # n_bands * 2 + 3
                nn.Conv2d(n_bands * 2 + 5, n_bands, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        elif dataset == 'Botswana':   # 145
            self.u2_channel = n_bands * 2 - 2
            self.conv3 = nn.Sequential(
                nn.Conv2d(n_bands * 2 + 5, n_bands, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        elif dataset == 'Urban':
            self.u2_channel = n_bands * 2
        elif dataset == 'DC':
            self.u2_channel = n_bands * 2 - 2
            self.conv3 = nn.Sequential(
                nn.Conv2d(n_bands * 2 + 5, n_bands, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        elif dataset == 'IndianP':
            self.u2_channel = n_bands * 2 - 2
        elif dataset == 'KSC':
            self.u2_channel = n_bands * 2 - 2
        elif dataset == 'Chikusei':
            self.u2_channel = n_bands * 2
        self.U2 = nn.Sequential(
            nn.Conv2d(self.u2_channel, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.convf2 = nn.Sequential(
            nn.Conv2d(self.n_bands*4 ,512,1,1,0),
            nn.ReLU(),
            nn.Conv2d(512,self.u2_channel,1,1,0),
            nn.ReLU(),
        )
        self.convTST1 = nn.Sequential(
            nn.Conv2d(1024,512,1,1,0),
            nn.ReLU(),
            nn.Conv2d(512,self.n_bands,1,1,0),
            nn.ReLU(),
        )
        self.convTST2 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(512, self.n_bands*2, 1, 1, 0),
            nn.ReLU(),
        )
        self.conve = nn.Conv2d(self.n_bands*2,self.n_bands,1,1,0)
        self.transformer1 = TransformerModel_rectangle(
            map_size=8,
            M_channel=n_bands * 2,
            dim=128,
            depth=5,
            heads=8,
            mlp_dim=n_bands,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
        )
        self.transformer2 = TransformerModel_rectangle(
            map_size=32,
            M_channel=n_bands,
            dim=64,
            depth=5,
            heads=8,
            mlp_dim=n_bands,
            dropout_rate=0.1,
            attn_dropout_rate=0.1
        )
        self.ca = ChannelAttention(2 * n_bands)
        self.ca1 = ChannelAttention(n_bands)
        self.sa = SpatialAttention()
        self.MS_MSA1 = MSAB(self.n_bands * 6, 1024, 1, dataset)  # MSA+PreNorm
        self.MS_MSA2 = MS_MSA(self.n_bands * 4, 1024, 1, dataset)
        self.MS_MSA1_TSTfirst = MS_MSA(self.n_bands * 2, 1024, 1, dataset)
        self.MS_MSA2_TSTfirst = MS_MSA2(self.n_bands * 4, 1024, 1, dataset)
        self.deconv1_TSTfirst = ConvMoudle1(self.n_bands * 6, self.n_bands)
        self.deconv2_TSTfirst = nn.Conv2d(self.n_bands * 3,self.u2_channel,1,1,0)
        if dataset == 'PaviaU':
            self.deconv1 = ConvMoudle1(1024, u1_channel)  # PU:618,urban
            self.deconv2 = ConvMoudle2_SK(self.u2_channel, self.u2_channel,self.n_bands,34)
            self.ScaleStrip = ScaleStrip(u1_channel, u1_channel, nn.BatchNorm2d)
            self.ScaleStrip2 = ScaleStrip(self.u2_channel, self.u2_channel, nn.BatchNorm2d)
        elif dataset == 'Urban':
            self.deconv1 = ConvMoudle1(972, u1_channel)  # PU:618,urban
            self.deconv2 = ConvMoudle2_SK(self.u2_channel, self.u2_channel,self.n_bands,27)
        elif dataset == 'Botswana':
            self.deconv1 = ConvMoudle1(1024, u1_channel)   # 870
            self.deconv2 = ConvMoudle2_SK(self.u2_channel, self.u2_channel, self.n_bands,36)
        elif dataset == 'DC':
            self.deconv1 = ConvMoudle1(1024 , u1_channel)  # 1146
            self.deconv2 = ConvMoudle2_SK(self.u2_channel, self.u2_channel, self.n_bands, 38)
        self.multi_scale_fusion = MultiScaleFusion_Pooling(u1_channel, pool_sizes, u1_channel)
        self.multi_scale_fusion2 = MultiScaleFusion_Pooling(self.u2_channel, pool_sizes, self.u2_channel)
        self.ScaleStrip = ScaleStrip(u1_channel, u1_channel, nn.BatchNorm2d)
        self.ScaleStrip2 = ScaleStrip(self.u2_channel, self.u2_channel, nn.BatchNorm2d)
    def lrhr_interpolate(self, x_lr, x_hr):
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        gap_bands = self.n_bands / (self.n_select_bands - 1.0)
        for i in range(0, self.n_select_bands - 1):
            x_lr[:, int(gap_bands * i), ::] = x_hr[:, i, ::]
        x_lr[:, int(self.n_bands - 1), ::] = x_hr[:, self.n_select_bands - 1, ::]
        return x_lr
    def spatial_edge(self, x):  # 计算图像的空间边缘。它通过计算图像在水平和垂直方向上相邻像素之间的差异来得到。
        edge1 = x[:, :, 0:x.size(2) - 1, :] - x[:, :, 1:x.size(2), :]
        edge2 = x[:, :, :, 0:x.size(3) - 1] - x[:, :, :, 1:x.size(3)]
        return edge1, edge2
    def spectral_edge(self, x):  # 计算图像的光谱边缘。它通过计算图像在频带方向上相邻像素之间的差异来得到。
        edge = x[:, 0:x.size(1) - 1, :, :] - x[:, 1:x.size(1), :, :]
        zero_pad = torch.zeros_like(x[:, -1:, :, :])  # 创建与输入张量最后一层相同形状的零值张量
        edge = torch.cat((edge, zero_pad), dim=1)  # 在边缘张量的末尾添加零值张量
        return edge
    def forward(self, x_lr, x_hr):  # 初版
        if self.arch in ('DRTnet', 'DRTnet_GSIS', 'no_contrast'):
            a = self.D1(x_hr)  # [1, 5, 128, 128]-->[1, 103, 32, 32]
            a = a * self.ca1(a)  # a与自身经过通道注意力机制（Channel Attention）得到的权重相乘。
            a = a * self.sa(a)  # SpatialAttention
            b = self.D2(a)
            b = b * self.ca(b)
            b = b * self.sa(b)  # torch.Size([1, 206, 8, 8])03.17
            c = self.D2(x_lr)  # 分辨率输入图像x_lr通过D2模块进行下采样得到c。 [1, 103, 32, 32]-->c[1, 206, 8, 8]
            c = c * self.ca(c)
            c = c * self.sa(c)
            d = F.interpolate(x_lr, scale_factor=4, mode='bilinear')  # 对x_lr进行双线性插值上采样得到d [1, 103, 128, 128]
            d = d * self.ca1(d)
            d = d * self.sa(d)
            transformer_results = self.transformer1(b, c)  # hr、lr  线性映射、位置编码、残差结构、注意力计算；返回一个字典
            e = transformer_results['z']  # 从transformer_results中获取变换后的特征图e。从字典中获取键为 'z' 的值，将其赋给变量 e
            f1 = torch.cat((torch.cat((b, c), 1), e), 1)  # 将b、c和e按通道维度拼接得到f1 [1,618,8,8] n_band*2*3
            f1 = self.MS_MSA1(f1)  # , self.n_bands*6)  # 0508添加encoder 下采样1/16 [1,618,8,8]
            f1 = self.deconv1(f1)  # 0419 去掉条带卷及 # 卷积改变通道数
            f1 = F.interpolate(f1, scale_factor=4, mode='bilinear')  # 对f1进行双线性插值上采样得到与高分辨率图像尺寸相同的特征图。
            f1_channel = f1.shape[1]
            f1 = self.U1(f1)  # 卷积 [1, 103, 32, 32]
            transformer_results1 = self.transformer2(a, x_lr)
            g = transformer_results1['z']  # 特征图
            f2 = torch.cat((torch.cat((a, x_lr), 1), g), 1)
            f2 = torch.cat((f2, f1), 1)  # f2与f1按通道维度拼接得到新的f2  torch.Size([1, 648, 32, 32])
            f2 = self.MS_MSA2(f2, self.n_bands * 4)  # urban 1,648,32,32
            f2 = self.convf2(f2)  # [1, 204, 32, 32]
            f2 = self.deconv2(f1, f2)  # 0419 去掉条带卷积  0518dwconv  0604 SK [1, 204, 32, 32]
            f2 = F.interpolate(f2, scale_factor=4, mode='bilinear')  # 双线性插值 四倍  # ([1, 204, 128, 128])
            f2 = self.U2(f2)  # [1, 103, 128, 128]
            x = torch.cat((f2, x_hr), 1)
            x = torch.cat((x, d), 1)
            x = self.conv3(x)
            x_spat = x + self.conv_spat(x)  # torch.Size([1, 103, 128, 128])
            spat_edge1, spat_edge2 = self.spatial_edge(x_spat)  # 计算x_spat在水平和垂直方向上的边缘
            x_spec = x_spat + self.conv_spec(x_spat)
            spec_edge = self.spectral_edge(x_spec)  # 计算x_spec在频带方向上的边缘
            x = x_spec
            x = x_spec
        return x, x_spec  # x, x_spat, x_spec, spat_edge1, spat_edge2, spec_edge
