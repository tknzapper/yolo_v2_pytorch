import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import config as cfg
from utils import *
from weight_loader import *


layer1 = [
    (32, 3, 1, 1),        # (out_channels, kernel_size)
    "M",
    (64, 3, 1, 1),
    "M",
    (128, 3, 1, 1),
    (64, 1, 1, 0),
    (128, 3, 1, 1),
    "M",
    (256, 3, 1, 1),
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    "M",
    (512, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),         # reorg layer
]

layer2 = [
    "M",
    (1024, 3, 1, 1),
    (512, 1, 1, 0),
    (1024, 3, 1, 1),
    (512, 1, 1, 0),
    (1024, 3, 1, 1)
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyReLu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leakyReLu(x)
        return x


class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.data.size()
        # x = x.view(B, C * 4, H // 2, W // 2).contiguous()
        ws = self.stride
        hs = self.stride
        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))

        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        B, C, H, W = x.data.size()
        x = nn.AvgPool2d(x, (H, W))
        x = x.view(B, C)

        return x

class Darknet19(nn.Module):
    def __init__(self, weights_file=None):
        super(Darknet19, self).__init__()
        self.in_channels = 3
        self.layer1 = self._make_layers(layer1)
        self.layer2 = self._make_layers(layer2)
        self.conv = nn.Conv2d(self.in_channels, 1000, kernel_size=1, stride=1)
        self.global_avgpool = GlobalAvgPool2d()
        self.softmax = nn.Softmax(dim=1)

        if not weights_file == None:
            self.load_weights(weights_file)
        else:
            self.initialize_weights()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.conv(x)
        x = self.global_avgpool(x)
        x = self.softmax(x)

        return x

    def _make_layers(self, architecture):
        layers = []

        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(self.in_channels, x[0], kernel_size=x[1], stride=x[2], padding=x[3])]
                self.in_channels = x[0]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

    def load_weights(self, file):
        weights_file = file
        assert len(torch.load(weights_file).keys()) == len(self.state_dict().keys())
        dic = {}
        for keys, values in zip(self.state_dict().keys(), torch.load(weights_file).values()):
            dic[keys] = values
        self.load_state_dict(dic)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Yolo_v2(nn.Module):
    def __init__(self, weights_file=None):
        super(Yolo_v2, self).__init__()
        darknet19 = Darknet19(weights_file)
        self.num_classes = num_classes
        self.anchors = anchor_box

        # darknet backbone
        self.conv1 = darknet19.layer1
        self.conv2 = darknet19.layer2
        # detection layers
        self.conv3 = nn.Sequential(
            CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1)
        )
        self.conv4 = nn.Sequential(
            CNNBlock(3072, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, (5 + self.num_classes) * len(self.anchors), kernel_size=1)
        )
        self.reorg = ReorgLayer()
        # initialize weight
        self.init_weight(self.conv3)
        self.init_weight(self.conv4)

    def forward(self, x):
        x = self.conv1(x)
        passthrough = self.reorg(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat([passthrough, x], dim=1)
        x = self.conv4(x)

        return x

    def init_weight(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Yolo_v2().to(device)
    # summary(model, (3, 416, 416))
    # net = Darknet19(pretrained=True)