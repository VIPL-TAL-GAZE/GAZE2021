import torch
import torch.nn as nn
import torch.nn.functional as F

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

    _to_degrees = 180. / np.pi

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
        return torch.mean(torch.acos(sim) * self._to_degrees)


def pitchyaw_to_vector(a):
    sin = torch.sin(a)
    cos = torch.cos(a)
    return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], dim=1)


def get_angular_errors(a, b):
    a = pitchyaw_to_vector(a)
    b = pitchyaw_to_vector(b)
    sim = F.cosine_similarity(a, b, dim=1, eps=1e-8)
    sim = F.hardtanh_(sim, min_val=-1 + 1e-8, max_val=1 - 1e-8)
    return torch.acos(sim) * 180. / np.pi


def get_rsn_loss(pos_scores, neg_scores, pos_gaze, neg_gaze, gaze_gt, delta=3.):
    """
    region selection network loss
    :param pos_scores: N x M x 1
    :param neg_scores: N x M x 1
    :param pos_error: 1
    :param neg_error: 1
    :param delta:
    :return:
    """
    N = pos_scores.shape[0]
    prob_pos = pos_scores.view(N, -1).prod(dim=-1)
    prob_neg = neg_scores.view(N, -1).prod(dim=-1)

    prob_ratio = prob_pos / prob_neg

    delta = torch.tensor(delta, dtype=prob_ratio.dtype, device=prob_ratio.device)
    prob_ratio = torch.min(prob_ratio, delta)

    pos_error = get_angular_errors(pos_gaze, gaze_gt)
    neg_error = get_angular_errors(neg_gaze, gaze_gt)
    error_ratio = neg_error / pos_error
    rsn_loss = F.l1_loss(prob_ratio, error_ratio)
    return rsn_loss


def get_ohem_loss(pred, gt, keep_num=20):
    batch_size = pred.shape[0]
    ohem_l1 = F.l1_loss(pred, gt, reduction='none').sum(dim=1)

    sorted_ohem_l1, idx = torch.sort(ohem_l1, descending=True)
    keep_num = min(batch_size, keep_num)
    keep_idx = idx[:keep_num]

    ohem_l1 = ohem_l1[keep_idx]
    ohem_loss = ohem_l1.sum() / keep_num
    return ohem_loss, keep_idx


if __name__ == "__main__":

    criterion = get_loss('angular')

    a = torch.randn(8, 2)
    b = torch.randn(8, 2)

    loss = criterion(a, b)

    print(loss)

