import functools

import torch
from torch import nn
from torchvision.models import vgg19
from utils import adaIN


def wrap_reflection_pad(network):
    for name, m in network.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue

        if m.padding[1] == 0:
            continue

        x_pad = int(m.padding[1])
        m.padding = (0, 0)
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [network] + names[:-1])
        setattr(
            root, names[-1],
            nn.Sequential(
                nn.ReflectionPad2d((x_pad, x_pad, x_pad, x_pad)),
                m
            )
        )


class VGG19_Reflection_Encoder(nn.Module):
    BLOCKS = {
        'relu1_1': 1 + 1,
        'relu2_1': 6 + 1,
        'relu3_1': 11 + 1,
        'relu4_1': 20 + 1
    }

    def __init__(self):
        super().__init__()
        base = vgg19(pretrained=True)

        offset = 0
        self.feature_names = list(self.BLOCKS.keys())
        self.feature_extractor = nn.Module()
        for name, output_num_layer in self.BLOCKS.items():
            setattr(
                self.feature_extractor, 
                name,
                nn.Sequential(*base.features[offset:output_num_layer])
            )
            offset = output_num_layer
        self.feature_extractor.eval()
        self.feature_extractor.requires_grad_(False)
        wrap_reflection_pad(self)

    def forward(self, x):
        output = {}
        for name in self.feature_names:
            x = getattr(self.feature_extractor, name)(x)
            output[name] = x
        return output, x


class Reflection_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(512, 256, (3, 3), padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 128, (3, 3), padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 128, (3, 3), padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 64, (3, 3), padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 3, (3, 3), padding=1),
        )
        wrap_reflection_pad(self)

    def forward(self, x):
        return self.base(x)


class AdaINModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VGG19_Reflection_Encoder()
        self.decoder = Reflection_Decoder()

    def forward(self, images_content, images_style, alpha=1.0):
        _, feat_content = self.encoder(images_content)
        _, feat_style = self.encoder(images_style)
        t = adaIN(feat_content, feat_style)

        interpolate_t = alpha * t + (1. - alpha) * feat_content

        g_t = self.decoder(interpolate_t)
        return g_t
