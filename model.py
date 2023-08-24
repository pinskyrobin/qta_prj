"""
@author:     许峻玮
@date:       08/24/2023
"""

import math
import torch

from torch.nn.functional import avg_pool2d, relu
from torch.nn import BatchNorm2d, Conv2d, Linear, Module, Sequential


class Bottleneck(Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()

        self.bn1 = BatchNorm2d(in_channels)
        self.conv1 = Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)

        self.bn2 = BatchNorm2d(4 * growth_rate)
        self.conv2 = Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(relu(self.bn1(x)))
        out = self.conv2(relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = BatchNorm2d(in_channels)
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(relu(self.bn(x)))
        out = avg_pool2d(out, 2)
        return out


class DenseNet121(Module):
    def __init__(self, block=Bottleneck, growth_rate=32, reduction=0.5, num_classes=10):
        super(DenseNet121, self).__init__()
        self.growth_rate = growth_rate

        channels = 2 * growth_rate
        self.conv1 = Conv2d(3, channels, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, channels, 6)
        channels += 6 * growth_rate
        out_channels = int(math.floor(channels * reduction))
        self.trans1 = Transition(channels, out_channels)
        channels = out_channels

        self.dense2 = self._make_dense_layers(block, channels, 12)
        channels += 12 * growth_rate
        out_channels = int(math.floor(channels * reduction))
        self.trans2 = Transition(channels, out_channels)
        channels = out_channels

        self.dense3 = self._make_dense_layers(block, channels, 24)
        channels += 24 * growth_rate
        out_channels = int(math.floor(channels * reduction))
        self.trans3 = Transition(channels, out_channels)
        channels = out_channels

        self.dense4 = self._make_dense_layers(block, channels, 16)
        channels += 16 * growth_rate

        self.bn = BatchNorm2d(channels)
        self.linear = Linear(channels, num_classes)

    def _make_dense_layers(self, block, in_channels, n_blocks):
        layers = []

        for i in range(n_blocks):
            layers.append(block(in_channels, self.growth_rate))
            in_channels += self.growth_rate

        return Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)

        out = avg_pool2d(relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
