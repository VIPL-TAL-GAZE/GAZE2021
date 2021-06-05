import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import timm

from .gazenet import get_backbone
from models.backbone import *


class ITracker(nn.Module):
    def __init__(self, pretrained=True):
        super(ITracker, self).__init__()

        self.face_backbone = resnet50(pretrained=pretrained)
        self.leye_backbone = resnet50(pretrained=pretrained, replace_stride_with_dilation=[True, True, True])
        self.reye_backbone = resnet50(pretrained=pretrained, replace_stride_with_dilation=[True, True, True])

        self.fc_eye = nn.Sequential(
            nn.Linear(2048 * 2, 128),
            nn.ReLU(True)
        )
        self.fc_face = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(True),
            nn.Linear(128, 2)
        )

    def encode_input(self, data):
        face_data = data['face']
        leye_box = data['left_eye_box']
        reye_box = data['right_eye_box']

        B = face_data.shape[0]
        batch_order = torch.arange(B, dtype=leye_box.dtype, device=leye_box.device).view(B, 1)

        leye_box_ = torch.cat([batch_order, leye_box], dim=1)
        reye_box_ = torch.cat([batch_order, reye_box], dim=1)

        leye_data = torchvision.ops.roi_align(face_data, leye_box_, 128, aligned=True)
        reye_data = torchvision.ops.roi_align(face_data, reye_box_, 128, aligned=True)

        encoded_data = {
            'face': face_data,
            'left_eye': leye_data.clone(),
            'right_eye': reye_data.clone()
        }

        return encoded_data

    def forward(self, data):
        data = self.encode_input(data)

        face = data['face']
        left_eye = data['left_eye']
        right_eye = data['right_eye']

        B = face.shape[0]
        x_leye = self.leye_backbone(left_eye).view(B, -1)
        x_reye = self.reye_backbone(right_eye).view(B, -1)
        x_eye = torch.cat([x_leye, x_reye], dim=1)
        x_eye = self.fc_eye(x_eye)

        x_face = self.face_backbone(face).view(B, -1)
        x_face = self.fc_face(x_face)

        x = torch.cat([x_eye, x_face], dim=1)
        x = self.fc_out(x)
        return x


class AttBlock(nn.Module):
    def __init__(self, q_dim, kv_dim, d_k, qkv_bias=True):
        super(AttBlock, self).__init__()
        self.scale = d_k
        self.Wq = nn.Linear(q_dim, d_k, bias=qkv_bias)
        self.Wk = nn.Linear(kv_dim, d_k, bias=qkv_bias)
        self.Wv = nn.Linear(kv_dim, d_k, bias=qkv_bias)
        self.proj = nn.Linear(d_k, kv_dim)

    def forward(self, x_q, x_kv):
        q = self.Wq(x_q)
        k = self.Wk(x_kv)
        v = self.Wv(x_kv)

        # attn: b, s, s
        scores = torch.matmul(q.view(-1, self.scale, 1), k.view(-1, 1, self.scale))
        attn = F.softmax(scores / self.scale, dim=-1)
        x = torch.matmul(attn, v.view(-1, self.scale, 1)).view(-1, self.scale)
        x = self.proj(x)
        return x


class MultiHeadAttBlock(nn.Module):
    def __init__(self, features_dim, num_head, d_k, qkv_bias=True):
        super(MultiHeadAttBlock, self).__init__()

        self.dim = features_dim
        self.num_head = num_head
        self.d_k = d_k
        # assert head_dim * self.num_head == self.dim, "head num setting wrong"

        self.Wq = nn.Linear(self.dim, self.num_head * self.d_k, bias=qkv_bias)
        self.Wk = nn.Linear(self.dim, self.num_head * self.d_k, bias=qkv_bias)
        self.Wv = nn.Linear(self.dim, self.num_head * self.d_k, bias=qkv_bias)

        self.proj = nn.Linear(self.num_head * self.d_k, self.dim)

    def forward(self, x):
        # x: b, s, c
        B, S, C = x.shape

        # qkv: b, nhead, s, d_k
        q = self.Wq(x).view(B, S, self.num_head, self.d_k).transpose(1, 2)
        k = self.Wk(x).view(B, S, self.num_head, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(B, S, self.num_head, self.d_k).transpose(1, 2)

        # scores: b, nhead, s, s
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)

        # x_attn: b, nhead, s, d_k
        x_attn = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, self.num_head * self.d_k)
        output = self.proj(x_attn)
        return output


class ITrackerAttention(nn.Module):
    def __init__(self, pretrained=True):
        super(ITrackerAttention, self).__init__()

        self.face_backbone = resnet50(pretrained=pretrained)
        self.leye_backbone = resnet50(pretrained=pretrained, replace_stride_with_dilation=[True, True, True])
        self.reye_backbone = resnet50(pretrained=pretrained, replace_stride_with_dilation=[True, True, True])

        self.attn_l = AttBlock(q_dim=2048, kv_dim=2048, d_k=1024)
        self.attn_r = AttBlock(q_dim=2048, kv_dim=2048, d_k=1024)
        self.mlp = nn.Sequential(
            nn.Linear(2048 * 2, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True)
        )

        self.fc_out = nn.Linear(128, 2)

    def encode_input(self, data):
        face_data = data['face']
        leye_box = data['left_eye_box']
        reye_box = data['right_eye_box']

        B = face_data.shape[0]
        batch_order = torch.arange(B, dtype=leye_box.dtype, device=leye_box.device).view(B, 1)

        leye_box_ = torch.cat([batch_order, leye_box], dim=1)
        reye_box_ = torch.cat([batch_order, reye_box], dim=1)

        leye_data = torchvision.ops.roi_align(face_data, leye_box_, 64, aligned=True)
        reye_data = torchvision.ops.roi_align(face_data, reye_box_, 64, aligned=True)

        encoded_data = {
            'face': face_data,
            'left_eye': leye_data.clone(),
            'right_eye': reye_data.clone()
        }

        return encoded_data

    def forward(self, data):
        data = self.encode_input(data)

        face = data['face']
        left_eye = data['left_eye']
        right_eye = data['right_eye']

        B = face.shape[0]
        x_leye = self.leye_backbone(left_eye).view(B, -1)
        x_reye = self.reye_backbone(right_eye).view(B, -1)
        x_face = self.face_backbone(face).view(B, -1)

        x_leye = self.attn_l(x_q=x_face, x_kv=x_leye)
        x_reye = self.attn_r(x_q=x_face, x_kv=x_reye)
        x = torch.cat([x_leye, x_reye], dim=1)
        x = self.mlp(x)
        x = self.fc_out(x)

        return x


# class TBasicLayer(nn.Module):
#     def __init__(self, features_dim, out_dim, num_head, d_k, qkv_bias=True):
#         super(TBasicLayer, self).__init__()
#
#         self.mh = MultiHeadAttBlock(features_dim=features_dim, num_head=num_head, d_k=d_k, qkv_bias=qkv_bias)
#         self.norm = nn.LayerNorm(3)
#         self.fnn = nn.Sequential(
#             nn.Linear(features_dim, features_dim),
#             nn.ReLU(True),
#             nn.Linear(features_dim, out_dim)
#         )


class ITrackerMultiHeadAttention(nn.Module):
    def __init__(self, pretrained=True):
        super(ITrackerMultiHeadAttention, self).__init__()

        # feature extract
        self.face_backbone = resnet50(pretrained=pretrained)
        self.leye_backbone = resnet50(pretrained=pretrained, replace_stride_with_dilation=[True, True, True])
        self.reye_backbone = resnet50(pretrained=pretrained, replace_stride_with_dilation=[True, True, True])

        # multi-head attention
        self.mha = MultiHeadAttBlock(
            features_dim=2048,
            num_head=4,
            d_k=256
        )
        self.norm1 = nn.LayerNorm(2048)
        self.ffn = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048)
        )
        self.norm2 = nn.LayerNorm(2048)

        # fc output
        self.fc_eye = nn.Sequential(
            nn.Linear(2048 * 2, 128),
            nn.ReLU(True)
        )
        self.fc_face = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(True),
            nn.Linear(128, 2)
        )

    def encode_input(self, data):
        face_data = data['face']
        leye_box = data['left_eye_box']
        reye_box = data['right_eye_box']

        B = face_data.shape[0]
        batch_order = torch.arange(B, dtype=leye_box.dtype, device=leye_box.device).view(B, 1)

        leye_box_ = torch.cat([batch_order, leye_box], dim=1)
        reye_box_ = torch.cat([batch_order, reye_box], dim=1)

        leye_data = torchvision.ops.roi_align(face_data, leye_box_, 128, aligned=True)
        reye_data = torchvision.ops.roi_align(face_data, reye_box_, 128, aligned=True)

        encoded_data = {
            'face': face_data,
            'left_eye': leye_data.clone(),
            'right_eye': reye_data.clone()
        }

        return encoded_data

    def forward(self, data):
        data = self.encode_input(data)

        face = data['face']
        left_eye = data['left_eye']
        right_eye = data['right_eye']

        B = face.shape[0]
        x_leye = self.leye_backbone(left_eye).view(B, 1, -1)
        x_reye = self.reye_backbone(right_eye).view(B, 1, -1)
        x_face = self.face_backbone(face).view(B, 1, -1)

        x_seq = torch.cat([x_leye, x_reye, x_face], dim=1)
        x_seq = x_seq + self.norm1(self.mha(x_seq))
        x_ffn = x_seq + self.norm2(self.ffn(x_seq))
        x_leye, x_reye, x_face = torch.unbind(x_ffn, dim=1)

        x_eye = torch.cat([x_leye, x_reye], dim=1)
        x_eye = self.fc_eye(x_eye)

        x_face = self.fc_face(x_face)

        x = torch.cat([x_eye, x_face], dim=1)
        x = self.fc_out(x)

        return x
