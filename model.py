import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import seaborn as sns
import math
import operator
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet, self).__init__()
        layers=[3,4,6,3]
        self.in_channels = 32
        self.conv = nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.layer1 = self.make_layer(ResNetBlock, 32, layers[0], stride=1)
        self.layer2 = self.make_layer(ResNetBlock, 64, layers[1], stride=2)
        self.layer3 = self.make_layer(ResNetBlock, 128, layers[2], stride=2)
        self.layer4 = self.make_layer(ResNetBlock, 256, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape)
        x=x.unsqueeze(1)
        #print(x.shape)
        out = F.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out.squeeze()
