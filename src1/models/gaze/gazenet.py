import sys
sys.path.append('./')


import torch
import torch.nn as nn


from models.backbone import *
from timm.models import create_model


class GazeNet(nn.Module):
    def __init__(self, backbone='hrnet_w64', pretrained=True):
        super(GazeNet, self).__init__()

        self.backbone = create_model(
            backbone,
            pretrained=pretrained
        )

        self.gaze_fc = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.Linear(1000, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 2),
        )


    def forward(self, x):
        x = self.backbone(x)
        gaze = self.gaze_fc(x)

        return gaze


class BotNet(nn.Module):
    def __init__(self):
        super(BotNet, self).__init__()
        self.backbone = botnet()

        self.gaze_fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 2),
        )

    def forward(self, face):
        x = self.backbone(face)
        x = torch.flatten(x, 1)
        gaze = self.gaze_fc(x)

        return gaze


def get_model(name, **kwargs):

    if name == 'botnet':
        model = BotNet()
    else:
        model = GazeNet(backbone=name, **kwargs)

    return model



if __name__ == "__main__":

    # model = GazeNet()
    # model = get_model('botnet')
    model = get_model('hrnet_w64')
    # print(model)

    x = torch.randn(8, 3, 224, 224)
    outs = model(x)
    print(outs.shape)
