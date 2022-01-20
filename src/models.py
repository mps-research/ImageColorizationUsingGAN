import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, normalize, activation_func):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not normalize)
        self.bn = nn.BatchNorm2d(out_channels) if normalize else None
        self.af = activation_func

    def forward(self, x):
        y = self.conv(x)
        if self.bn:
            y = self.bn(y)
        y = self.af(y)
        return y


class Deconv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,  padding, normalize, activation_func):
        super(Deconv2dBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=not normalize)
        self.bn = nn.BatchNorm2d(out_channels) if normalize else None
        self.af = activation_func

    def forward(self, x):
        y = self.deconv(x)
        if self.bn:
            y = self.bn(y)
        y = self.af(y)
        return y


class Generator(nn.Module):
    def __init__(self, encoder, decoder):
        super(Generator, self).__init__()
        self.encoder = nn.ModuleList([Conv2dBlock(**b) for b in encoder])
        self.decoder = nn.ModuleList([Deconv2dBlock(**b) for b in decoder])

    def forward(self, x):
        encoder_outs = []
        for l in self.encoder:
            if not encoder_outs:
                encoder_outs.append(l(x))
            else:
                encoder_outs.append(l(encoder_outs[-1]))
        y = encoder_outs.pop(-1)
        for i, l in enumerate(self.decoder):
            y = l(y)
            if encoder_outs:
                y = torch.cat([y, encoder_outs.pop(-1)], dim=1)
        return y


class Discriminator(nn.Module):
    def __init__(self, blocks):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(*[Conv2dBlock(**b) for b in blocks])

    def forward(self, x):
        return self.net(x)
