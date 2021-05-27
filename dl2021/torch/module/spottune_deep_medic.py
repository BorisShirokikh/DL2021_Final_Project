import os
from copy import deepcopy

import torch.nn as nn
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

        self.policy_shape = 12
        self.policy_tracker = torch.zeros(self.policy_shape)
        self.policy_tracker_temp = torch.zeros(self.policy_shape)
        self.iter_tracker = 0
        self.iter_tracker_temp = 0

        self.parallelized_blocks = (nn.Conv3d, ResBlock3d, PreActivation3d)
        self.val_flag = False
        self.val_policy_tracker = torch.zeros(self.policy_shape)
        self.val_iter_tracker = 0

        self.branch1 = nn.Sequential(
            CenteredCrop(np.array([16, 16, 16])),
            PreActivation3d(n_chans_in, n1, kernel_size=3), #1
            PreActivation3d(n1, n1, kernel_size=3), #2
            ResBlock3d(n1, n2, kernel_size=3), #3
            ResBlock3d(n2, n2, kernel_size=3), #4
            ResBlock3d(n2, n3, kernel_size=3) #5
        )

        self.branch1_freezed = deepcopy(self.branch1)

        self.branch2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=3),
            PreActivation3d(n_chans_in, n1, kernel_size=3), #6
            PreActivation3d(n1, n1, kernel_size=3), #7
            ResBlock3d(n1, n2, kernel_size=3), #8
            ResBlock3d(n2, n2, kernel_size=3), #9
            ResBlock3d(n2, n3, kernel_size=3) #10
        )
        self.branch2_freezed = deepcopy(self.branch2)

        self.up3 = nn.Upsample(scale_factor=3, mode='trilinear', align_corners=False) if up else repeat

        self.fc = nn.Sequential(
            ResBlock3d(2*n3, 3*n3, kernel_size=1), #11
            nn.Conv3d(3*n3, n_chans_out, 1) #12
        )

        self.fc_freezed = deepcopy(self.fc)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_block(self, x, block_ft, block_fr, action_mask, i):

        for layer_ft, layer_fr in zip(block_ft, block_fr):
            policy_current = action_mask[..., i]
            x = layer_ft(x)*(1-policy_current) + layer_fr(x)*policy_current
            if isinstance(layer_ft, self.parallelized_blocks):
                i += 1
        return x, i

    def forward(self, inputs, policy):

        action = policy.contiguous()  # [12, 16]
        action_mask = action.view(-1, 1, 1, 1, action.shape[1])  # [12, 1, 1, 1, 16]
        i = 0

        x1, i = self.forward_block(inputs, self.branch1, self.branch1_freezed, action_mask, i)

        x2, i = self.forward_block(inputs, self.branch2, self.branch2_freezed, action_mask, i)

        x2 = self.up3(x2)
        x = torch.cat([x1, x2], 1)

        x, i = self.forward_block(x, self.fc, self.fc_freezed, action_mask, i)

        if self.val_flag:
            self.val_iter_tracker += action.shape[0]  # batch size
            self.val_policy_tracker += torch.sum(action, dim=0).to('cpu')
        else:
            self.iter_tracker += action.shape[0]  # batch size
            self.policy_tracker += torch.sum(action, dim=0).to('cpu')

        return x

    def save_policy(self, folder_name):

        we_are_here = os.path.abspath('.')
        folder_to_store_in = os.path.join(we_are_here, folder_name)
        if not os.path.exists(folder_to_store_in):
            os.mkdir(folder_to_store_in)

        torch.save(self.policy_tracker, os.path.join(folder_to_store_in, 'policy_record'))

        f = open(os.path.join(folder_to_store_in, 'iter_record'), "w")
        f.write(str(self.iter_tracker))
        f.close()

        self.policy_tracker = torch.zeros(self.policy_shape)
        self.iter_tracker = 0

    def get_val_stats(self):

        val_stats = self.val_policy_tracker.detach().numpy() / self.val_iter_tracker

        self.val_policy_tracker = torch.zeros(self.policy_shape)
        self.val_iter_tracker = 0

        tb_record = {}
        for i in range(self.policy_shape-3):
            tb_record['val: block ' + str(i+1)] = val_stats[i]
        for i in range(self.policy_shape-3, self.policy_shape):
            tb_record['val: shortcut ' + str(i-(self.policy_shape-4))] = val_stats[i]

        return tb_record

    def get_train_stats(self):

        train_stats = (self.policy_tracker.detach().numpy() - self.policy_tracker_temp.detach().numpy()) / \
                      (self.iter_tracker - self.iter_tracker_temp)

        self.iter_tracker_temp = self.iter_tracker
        self.policy_tracker_temp = self.policy_tracker.clone()

        tb_record = {}
        for i in range(self.policy_shape-3):
            tb_record['train: block ' + str(i+1)] = train_stats[i]
        for i in range(self.policy_shape-3, self.policy_shape):
            tb_record['train: shortcut ' + str(i-(self.policy_shape-4))] = train_stats[i]

        return tb_record