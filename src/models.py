from torch import nn
import torch
import numpy as np


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=None):
        super(UNetDown, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 4, 2, 1, bias=True)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # if dropout:
        #    layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.Conv2d(in_size, 4 * out_size, 3, 1, 1),

            nn.BatchNorm2d(4 * out_size, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(2),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class UNetUp_old(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=True),
            nn.Conv2d(out_size, out_size, 3, 1, 1),
            # nn.ReLU(inplace=True),  # paper
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)  # , dropout=0.5)
        self.down5 = UNetDown(512, 512)  # , dropout=0.5)
        self.down6 = UNetDown(512, 512)  # , dropout=0.5)
        self.down7 = UNetDown(512, 512)  # , dropout=0.5)
        # self.down8 = UNetDown(1024, 1024, dropout=0.5)
        #
        # self.up1 = UNetUp(1024, 1024, dropout=0.5)
        self.up2 = UNetUp(512, 512)  # , dropout=0.5)
        self.up3 = UNetUp(1024, 512)  # , dropout=0.5)
        self.up4 = UNetUp(1024, 512)  # , dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Conv2d(128, 4 * out_channels, 3, 1, 1),
            nn.PixelShuffle(2),
            # nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            # nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        # d8 = self.down8(d7)
        # u1 = self.up1(d8, d7)
        u2 = self.up2(d7, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, img_size=(256, 256), patch_size=(16, 16), in_channels=1):
        super(Discriminator, self).__init__()
        print('patch size ', patch_size)

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]

            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        if img_size[0] % patch_size[0] or img_size[1] % patch_size[1]:
            assert 'Can\'t do patch from such image size.'

        if img_size[0] / patch_size[0] == img_size[1] / patch_size[1]:
            assert 'Can\'t do patch from such image size.'

        k = int(np.log2(img_size[0] / patch_size[0]))
        model_layers = [*discriminator_block(in_channels * 2, 64)]

        for i in range(k - 1):
            model_layers += discriminator_block(64 * 2 ** i, 64 * 2 ** (i + 1))
        print('layers D', k)
        # print(model_layers)

        self.model = nn.Sequential(

            *model_layers,
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64 * 2 ** (i + 1), 1, 4, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, condition, target):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((condition, target), 1)
        return self.model(img_input)
