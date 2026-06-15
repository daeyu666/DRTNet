import torch
import torch.nn as nn
import torch.nn.functional as F
# from .basic_blocks  import *
from models.basic_blocks import *
import numpy as np
import cv2
import math
from models.Transformer import TransformerModel
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d




# 通道注意力模块
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


# 空间注意力模块
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


##################################
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp

        self.deconv1 = nn.Conv2d(  # 条带卷积
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )

        self.bn2 = BatchNorm(in_channels // 4 + in_channels // 4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x, inp = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.h_transform(x)
        x3 = self.deconv3(x3)
        x3 = self.inv_h_transform(x3)
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
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
            elif isinstance(m, nn.BatchNorm2d): #
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        # h_transform 函数对输入特征进行了一个水平方向的变换（或者说扩展），然后再通过 self.deconv3 执行卷积操作。
        # h_transform 函数实际上是对输入特征在水平方向上进行了一定的操作，相当于引入了横向上的信息交互，
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))  # # 在特征图的右侧填充与宽度相同大小的0，实际上就是在水平方向上扩展了一倍
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]  # # 将特征图 reshape 成三维张量，最后一维是原宽度的两倍
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)  # 再将特征图 reshape 回原始形状，但是宽度变成了原来的两倍减一
        return x

    def inv_h_transform(self, x):  # 对h_transform后的张量进行恢复
        shape = x.size()
        # # 将输入特征图x按照给定的形状重新排列，其中-1表示自动计算该维度的大小，.contiguous()表示返回一个内存连续的tensor。
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        # 对排列后的特征图进行填充操作，沿着最后两个维度分别在右侧和下侧填充0
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        # 取得到的特征图的子区域，即沿着最后一个维度取从0到shape[-2]（不包括shape[-2]）的部分
        x = x[..., 0: shape[-2]]
        return x


    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

#################################################################
class MCT(nn.Module):
    def __init__(self,
                 arch,
                 scale_ratio,
                 n_select_bands,
                 n_bands,
                 dataset=None,
                 # u1_channel

                 ):
        """Load the pretrained ResNet and replace top fc layer."""
        super(MCT, self).__init__()

        self.scale_ratio = scale_ratio
        self.n_bands = n_bands  # 103
        self.arch = arch
        self.n_select_bands = n_select_bands # 5
        self.weight = nn.Parameter(torch.tensor([0.5]))

        u1_channel = 484  # KSC 528，532 # PaviaU308
        self.u2_channel = 324

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
            nn.Conv2d(156, n_bands*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands*2, n_bands*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        #Pavia(n_bands*3-2) PaviaU(n_bands*3-1)

        # #根据数据集的类型设置了 U1 模块的通道数。
        # 根据不同的数据集类型，u1_channel 的值会有所不同。U1 模块包含多个卷积层和ReLU激活函数，用于处理输入特征图。
        # u1_channel = 0
        # global u1_channel
        if dataset == 'Pavia':
            n_bands = n_bands*3-2
            #self.u1_channel = n_bands*3-2
        elif dataset == 'PaviaU':
            #n_bands = n_bands * 3 - 1
            u1_channel = n_bands * 3 - 1  # 308
        elif dataset == 'Botswana':
            u1_channel = n_bands * 3 - 3
        elif dataset == 'Urban':
            u1_channel = n_bands * 3 - 2
        elif dataset == 'Washington':
            u1_channel = n_bands * 3-1
        # n_bands = u1_channel
        self.U1 = nn.Sequential(
            nn.Conv2d(u1_channel, n_bands*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands*2, n_bands*1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Pavia(n_bands*2) PaviaU(n_bands*2-2)
        # u2_channel = 0
        if dataset == 'Pavia':
            self.u2_channel = n_bands * 2
        elif dataset == 'PaviaU':
            self.u2_channel = n_bands * 2 - 2 # 204
        elif dataset == 'Botswana':
            self.u2_channel = n_bands * 2 - 2
        elif dataset == 'Urban':
            self.u2_channel = n_bands * 2
        elif dataset == 'Washington':
            self.u2_channel = n_bands*2-2
        self.U2 = nn.Sequential(
            #nn.Conv2d( u2_channel, n_bands, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.u2_channel, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(n_bands*2+5, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.transformer1 = TransformerModel(
            map_size=8,
            M_channel = n_bands*2,
            dim=128,
            depth=5,
            heads=8,
            mlp_dim=n_bands,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
        )
        self.transformer2 = TransformerModel(
            map_size = 32,
            M_channel=n_bands,
            dim=64,
            depth=5,
            heads=8,
            mlp_dim=n_bands,
            dropout_rate=0.1,
            attn_dropout_rate=0.1
        )

        self.ca = ChannelAttention(2 * n_bands)
        self.ca1 = ChannelAttention( n_bands)
        self.sa = SpatialAttention()
        self.decoder1 = DecoderBlock(n_bands*6, n_bands*6, nn.BatchNorm2d)  # 条带卷积
        self.decoder2 = DecoderBlock(n_bands * 4, n_bands * 4, nn.BatchNorm2d)
        self.decoder3 = DecoderBlock(n_bands * 1, n_bands * 1, nn.BatchNorm2d)# 没用到


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

        return edge


    def forward(self, x_lr, x_hr):

        if self.arch == 'MCT':
            # 高分辨率输入图像下采样4次--> ChannelAttention--> SpatialAttention
            a = self.D1(x_hr)
            a = a * self.ca1(a)  # a与自身经过通道注意力机制（Channel Attention）得到的权重相乘。
            a = a*self.sa(a)  # SpatialAttention

            b = self.D2(a)
            b = b * self.ca(b)
            b = b * self.sa(b)

            c = self.D2(x_lr)  # 分辨率输入图像x_lr通过D2模块进行下采样得到c。
            c = c * self.ca(c)
            c = c * self.sa(c)

            d = F.interpolate(x_lr, scale_factor=4, mode='bilinear')  # 对x_lr进行双线性插值上采样得到d
            d = d * self.ca1(d)
            d = d * self.sa(d)
#######################################################################################
            transformer_results = self.transformer1(b, c)  # hr、lr  线性映射、位置编码、残差结构、注意力计算；返回一个字典
            e = transformer_results['z']  # 从transformer_results中获取变换后的特征图e。从字典中获取键为 'z' 的值，将其赋给变量 e
            f1 = torch.cat((torch.cat((b,c), 1), e),1)  # 将b、c和e按通道维度拼接得到f1
            f1 = self.decoder1(f1)  # f1通过解码器decoder1进行解码。条带卷积
            f1 = F.interpolate(f1, scale_factor=4, mode='bilinear')  # 对f1进行双线性插值上采样得到与高分辨率图像尺寸相同的特征图。
            f1_channel = f1.shape[1]
            f1 = self.U1(f1)  # 卷积
 ###################################################################################
            transformer_results1 = self.transformer2(a,x_lr)
            g = transformer_results1['z']  # 特征图
            f2 = torch.cat((torch.cat((a,x_lr),1),g),1)
            f2 = torch.cat((f2,f1),1)  # f2与f1按通道维度拼接得到新的f2


            f2 = self.decoder2(f2)
            f2 = F.interpolate(f2, scale_factor=4, mode='bilinear')

            f2 = self.U2(f2)

            x = torch.cat((f2, x_hr), 1)
            x = torch.cat((x, d), 1)
            x = self.conv3(x)
            x_spat = x + self.conv_spat(x)
            spat_edge1, spat_edge2 = self.spatial_edge(x_spat)  # 计算x_spat在水平和垂直方向上的边缘

            x_spec = x_spat + self.conv_spec(x_spat)
            spec_edge = self.spectral_edge(x_spec)  # 计算x_spec在频带方向上的边缘

            x = x_spec
        return x, x_spat, x_spec, spat_edge1, spat_edge2, spec_edge
