import sys
sys.path.append('./')


import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import timm


from models.backbone import *


def get_backbone(name, **kwargs):
    models = {
        # resnet
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
        'resnext50_32x4d': resnext50_32x4d,
        'resnext101_32x8d': resnext101_32x8d,
        'wide_resnet50_2': wide_resnet50_2,
        'wide_resnet101_2': wide_resnet101_2,
        # densenet
        'densenet121': densenet121,
        'densenet169': densenet169,
        'densenet201': densenet201,
        'densenet161': densenet161,
        # gaze360
        'efficientnet_b8': efficientnet_b8,
        'swin_small': swin_small,
        'swin_large': swin_large,
        'deit_base': deit_base,
        'deit_small': deit_small,
        'deit_tiny': deit_tiny,
        'resnest269e': resnest269e,
        'mobilenetv2_100': mobilenetv2_100,
        'mobilenetv2_140': mobilenetv2_140,
        'mobilenetv3_large_075': mobilenetv3_large_075,
        'mobilenetv3_large_100': mobilenetv3_large_100,
        'mobilenetv3_rw': mobilenetv3_rw
        }
    name = name.lower()
    if name not in models:
        raise ValueError('%s\n\t%s' % (str(name), '\n\t'.join(sorted(models.keys()))))
    net = models[name](**kwargs)



    return net


def get_embedding_dim(backbone):
    embbeding_num = 0

    # bb = backbone_i[1]
    if backbone in ['resnet18', 'resnet34', 'iresnet_se50']:
        embbeding_num += 512
    elif backbone in ['densenet121']:
        embbeding_num += 1024
    elif backbone in ['densenet169']:
        embbeding_num += 1664
    elif backbone in ['densenet201']:
        embbeding_num += 1920
    elif backbone in ['densenet161']:
        embbeding_num += 2208
    elif backbone in ['gridnet']:
        embbeding_num += 1250
    elif backbone in ['efficientnet_b8']:
        embbeding_num += 2816
    elif backbone in ['efficientnet_b0']:
        embbeding_num += 1280
    elif backbone in ['efficientnet_el', 'swin_large']:
        embbeding_num += 1536
    elif backbone in ['swin_small', 'deit_base']:
        embbeding_num += 768
    elif backbone in ['deit_small']:
        embbeding_num += 384
    elif backbone in ['deit_tiny']:
        embbeding_num += 192


    return embbeding_num


def get_fc_layers(backbones, dropout=0.0):
    embbeding_num = 0
    for _, bb in backbones:
        # bb = backbone_i[1]
        if bb in ['resnet18', 'resnet34', 'iresnet_se50']:
            embbeding_num += 512
        elif bb in ['densenet121']:
            embbeding_num += 1024
        elif bb in ['densenet169']:
            embbeding_num += 1664
        elif bb in ['densenet201']:
            embbeding_num += 1920
        elif bb in ['densenet161']:
            embbeding_num += 2208
        elif bb in ['gridnet']:
            embbeding_num += 1250
        elif bb in ['efficientnet_b8']:
            embbeding_num += 2816
        elif bb in ['efficientnet_b0']:
            embbeding_num += 1280
        elif bb in ['efficientnet_el', 'swin_large']:
            embbeding_num += 1536
        elif bb in ['swin_small', 'deit_base']:
            embbeding_num += 768
        elif bb in ['deit_small']:
            embbeding_num += 384
        elif bb in ['deit_tiny']:
            embbeding_num += 192
        elif 'mobilenet' in bb:
            embbeding_num += 1280
        else:
            embbeding_num += 2048

    fc = nn.Sequential(
        nn.Linear(embbeding_num, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(128, 128),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(128, 2),
    )
    return fc


class GazeNet(nn.Module):
    def __init__(self, backbones=None, pretrained=False, dropout=0.0):
        super(GazeNet, self).__init__()

        if backbones is None:
            backbones = ['resnet50']

        self.bacbones = backbones

        self.module_dict = nn.ModuleDict()
        for name, bb in backbones:
            self.module_dict[name] = get_backbone(bb, pretrained=pretrained)

        self.gaze_fc = get_fc_layers(backbones, dropout=dropout)

        self.input_dict = {}

    def encode_input(self, data):
        face_data = data['face']
        # leye_box = list(torch.unbind(data['left_eye_box'], dim=0))
        # reye_box = list(torch.unbind(data['right_eye_box'], dim=0))
        # print(len(leye_box))
        # print(leye_box[0].shape)
        # leye_data = torchvision.ops.roi_align(face_data, leye_box, 68, aligned=True)
        # reye_data = torchvision.ops.roi_align(face_data, reye_box, 68, aligned=True)

        encoded_data = {
            'face': face_data,
            # 'left_eye': leye_data,
            # 'right_eye': reye_data
        }

        return encoded_data

    def forward(self, x):
        x = self.encode_input(x)
        fc_input = []
        for name, net in self.module_dict.items():
            feat = net(x[name])
            fc_input.append(feat)
        x = torch.cat(fc_input, dim=1)
        x = torch.flatten(x, 1)
        gaze = self.gaze_fc(x)

        return gaze




class GazeNetOrigin(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=False):
        super(GazeNetOrigin, self).__init__()

        self.backbone = get_backbone(backbone, pretrained=pretrained)

        # TODO; temp implementation, need modify
        if backbone in ['resnet18', 'resnet34', 'iresnet_se50']:
            self.gaze_fc = nn.Linear(512, 2)
        elif backbone == 'densenet121':
            self.gaze_fc = nn.Linear(1024, 2)
        elif backbone == 'densenet169':
            self.gaze_fc = nn.Linear(1664, 2)
        elif backbone == 'densenet201':
            self.gaze_fc = nn.Linear(1920, 2)
        elif backbone == 'densenet161':
            self.gaze_fc = nn.Linear(2208, 2)
        elif backbone in ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']:
            self.gaze_fc = nn.Linear(4096, 2)
        else:
            self.gaze_fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        gaze = self.gaze_fc(x)

        return gaze


if __name__ == "__main__":

    model = GazeNet(backbone='gaze360', pretrained=False)
    # print(model)

    x = torch.randn(8, 3, 224, 224)
    outs = model(x)

    print(outs)

