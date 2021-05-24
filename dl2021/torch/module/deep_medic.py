import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import numpy as np

from dpipe.layers.resblock import ResBlock3d
from dpipe.layers.conv import PreActivation3d
from dpipe.layers.structure import CenteredCrop

# adapt from https://github.com/Kamnitsask/deepmedic

def repeat(x, n=3):
    # nc333
    b, c, h, w, t = x.shape
    x = x.unsqueeze(5).unsqueeze(4).unsqueeze(3)
    x = x.repeat(1, 1, 1, n, 1, n, 1, n)
    return x.view(b, c, n*h, n*w, n*t)


class DeepMedic(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, n1=30, n2=40, n3=50, up=True):
        super(DeepMedic, self).__init__()
        #n1, n2, n3 = 30, 40, 50

        # n = 2*n3
        self.branch1 = nn.Sequential(
                CenteredCrop(np.array([16,16,16])),
                PreActivation3d(n_chans_in, n1, kernel_size=3),
                PreActivation3d(n1, n1, kernel_size=3),
                ResBlock3d(n1, n2, kernel_size=3),
                ResBlock3d(n2, n2, kernel_size=3),
                ResBlock3d(n2, n3, kernel_size=3))

        self.branch2 = nn.Sequential(
                nn.AvgPool3d(kernel_size=3),
                PreActivation3d(n_chans_in, n1, kernel_size=3),
                PreActivation3d(n1, n1, kernel_size=3),
                ResBlock3d(n1, n2, kernel_size=3),
                ResBlock3d(n2, n2, kernel_size=3),
                ResBlock3d(n2, n3, kernel_size=3))

        self.up3 = nn.Upsample(scale_factor=3,
                mode='trilinear', align_corners=False) if up else repeat

        self.fc = nn.Sequential(
                ResBlock3d(2*n3, 3*n3, kernel_size=1),
                nn.Conv3d(3*n3, n_chans_out, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        # print(inputs.shape, low_input.shape)
        # norm_input, low_input = inputs
        print(inputs.shape)
        x1 = self.branch1(inputs)
        x2 = self.branch2(inputs)
        # print(x1.shape, x2.shape)
        x2 = self.up3(x2)
        # print(x2.shape)
        x = torch.cat([x1, x2], 1)
        x = self.fc(x)
        # print(x.shape)
        return x

