import torch
import torch.nn as nn


class SKConv(nn.Module):
    def __init__(self, inplanes, planes, groups=1, stride=1, M=2, r=16, L=32):
        super(SKConv, self).__init__()
        self.M = M
        self.inplanes = inplanes
        self.planes = planes
        d = max(inplanes // r, L)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(inplanes, d, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fcs = nn.ModuleList([
            nn.Conv2d(d, inplanes, 1, bias=False) for _ in range(M)
        ])
        self.softmax = nn.Softmax(dim=1)
        self.project = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        feats = torch.stack((x1, x2), dim=1)
        fused = feats.sum(dim=1)
        descriptor = self.avg_pool(fused)
        descriptor = self.fc(descriptor)
        attention = torch.cat([fc(descriptor) for fc in self.fcs], dim=1)
        attention = attention.view(batch_size, self.M, self.inplanes, 1, 1)
        attention = self.softmax(attention)
        out = (feats * attention).sum(dim=1)
        return self.project(out)
