import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from config import *
from utils import *


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

        # ws = self.stride
        # hs = self.stride
        # x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        # x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        # x = x.view(B, C, hs * ws, int(H /hs), int(W / ws)).transpose(1, 2).contiguous()
        # x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))

        x = x.view(B, C * 4, H // 2, W // 2).contiguous()

        return x


class Darknet19(nn.Module):
    def __init__(self):
        super(Darknet19, self).__init__()
        self.in_channels = 3
        self.layer1 = self._make_layers(layer1)
        self.layer2 = self._make_layers(layer2)
        self.init_weight(self.layer1)
        self.init_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

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


class Yolo_v2(nn.Module):
    def __init__(self, train=False):
        super(Yolo_v2, self).__init__()
        darknet19 = Darknet19()
        self.train = train
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
        # self.downsampler = nn.Sequential(
        #     CNNBlock(512, 64, kernel_size=1, stride=1)
        # )
        self.conv4 = nn.Sequential(
            CNNBlock(3072, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, (5 + self.num_classes) * len(self.anchors), kernel_size=1)
        )
        self.reorg = ReorgLayer()
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
    model = Yolo_v2().to(device)
    summary(model, (3, 416, 416))