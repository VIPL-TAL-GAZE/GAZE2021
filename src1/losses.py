import torch
import torch.nn as nn
import torch.nn.functional as F


import math
import numpy as np


def get_loss(loss_name):

    if loss_name == 'l1':
        return nn.L1Loss()
    elif loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'smoothl1':
        return nn.SmoothL1Loss()
    elif loss_name == 'angular':
        return AngularLoss()
    else:
        raise ValueError


class AngularLoss(nn.Module):

    # _to_degrees = 180. / np.pi

    def __init__(self):
        super(AngularLoss, self).__init__()

    def pitchyaw_to_vector(self, a):
        if a.shape[1] == 2:
            sin = torch.sin(a)
            cos = torch.cos(a)
            return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], dim=1)
        else:
            raise ValueError('Do not know how to convert tensor of size %s' % a.shape)

    def forward(self, a, b):
        a = self.pitchyaw_to_vector(a)
        b = self.pitchyaw_to_vector(b)
        sim = F.cosine_similarity(a, b, dim=1, eps=1e-8)
        sim = F.hardtanh_(sim, min_val=-1+1e-8, max_val=1-1e-8)
        # return torch.mean(torch.acos(sim) * self._to_degrees)

        return torch.mean(torch.acos(sim))


if __name__ == "__main__":

    # criterion = get_loss('angular')

    # a = torch.randn(8, 2)
    # b = torch.randn(8, 2)

    # loss = criterion(a, b)

    # print(loss)

    torch.manual_seed(1)

    # criterion = get_loss('eloss')
    criterion = get_loss('fareloss')

    gl = torch.randn(8, 3)
    gr = torch.randn(8, 3)
    pl = torch.randint(1, 9, (8,)) * 0.1
    pr = torch.randint(1, 9, (8,)) * 0.1
    g_target = torch.randn(8, 2)

    loss = criterion(gl, gr, pl, pr, g_target)

    print(loss)

