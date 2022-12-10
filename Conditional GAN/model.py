# Conditional GAN based on WGANGP

import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.conv_block(channels_img + 1, features_d, 4, 2, 1),
            self.conv_block(features_d, features_d * 2, 4, 2, 1),
            self.conv_block(features_d * 2, features_d * 4, 4, 2, 1),
            self.conv_block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )
        self.embed = nn.Embedding(num_classes, img_size * img_size)
        self.img_size = img_size

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        labels = self.embed(labels)
        labels = labels.view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, labels], dim=1)
        return self.crit(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, num_classes, img_size, embed_dim):
        super(Generator, self).__init__()
        self.genrtr = nn.Sequential(
            self.deconv_block(z_dim + embed_dim, features_g * 16, 4, 1, 0),
            self.deconv_block(features_g * 16, features_g * 8, 4, 2, 1),
            self.deconv_block(features_g * 8, features_g * 4, 4, 2, 1),
            self.deconv_block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.img_size = img_size

    def forward(self, x, labels):
        labels = self.embed(labels)
        labels = labels.view(labels.shape[0], labels.shape[1], 1, 1) # B x E x 1 x 1
        # print(labels.shape[0], x.shape[0])
        x = torch.cat([x, labels], dim=1) # B x (Z + E) x 1 x 1
        return self.genrtr(x)

    def deconv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)