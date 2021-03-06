import os
import json
import torch
import numpy as np


def save_json(fn, data):
    with open(fn, 'w') as f:
        json.dump(data, f)


def save_results(res):
    np.savetxt('within_eva_results.txt', res, delimiter=',')
    
    if os.path.exists('submission_within_eva.zip'):
        os.system('rm submission_within_eva.zip')

    os.makedirs('submission_within_eva')
    os.system('mv within_eva_results.txt ./submission_within_eva')
    os.system('zip -r submission_within_eva.zip submission_within_eva')
    os.system('rm -rf submission_within_eva')


def save_checkpoint(state, args, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    # directory = "runs/%s/%s/"%(args.backbone, args.checkname)
    directory = os.path.join(args.ckpt_root, args.backbone, args.checkname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # filename = directory + filename
    filename = os.path.join(directory, filename)
    torch.save(state, filename)


class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterTensor(torch.nn.Module):
    def __init__(self, dtype=torch.float):
        super(AverageMeterTensor, self).__init__()
        self.dtype = dtype
        self.reset()

    def reset(self):
        self.val = torch.tensor(0, dtype=self.dtype)
        self.avg = torch.tensor(0, dtype=self.dtype)
        self.sum = torch.tensor(0, dtype=self.dtype)
        self.count = torch.tensor(0, dtype=self.dtype)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pitchyaw_to_vector(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    Args:
        pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
    """
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def vector_to_pitchyaw(vectors):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.

    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * 180.0 / np.pi


def main_angular_error(gl, gr, pl, pr, g_target):

    gl = gl.cpu().data.numpy()
    gr = gr.cpu().data.numpy()
    pl = pl.cpu().data.numpy()
    pr = pr.cpu().data.numpy()
    g_target = g_target.cpu().data.numpy()

    g_target = pitchyaw_to_vector(g_target)
    el = angular_error(gl, g_target)
    er = angular_error(gr, g_target)

    error = (pl > pr) * el + (pl <= pr) * er

    return np.mean(error)
