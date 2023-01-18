# Cycle GAN Generator and Discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim_in, dim_out, 3),
            nn.InstanceNorm2d(dim_out),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim_out, dim_out, 3),
            nn.InstanceNorm2d(dim_out),
        )

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    