# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
# from SDNet_models.sdnet import DictConv2d

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x






class Model(nn.Module):
    def __init__(self, input_dim=256*256, target_shape=[1, 256, 256]):
        super().__init__()
        self.target_shape = target_shape
        self.hidden = hidden = 64

        self.net1 = nn.Sequential(
            nn.Linear(1601, 256*256),
        )

        self.A_down1 = UNetDown(1, 64, normalize=False)
        self.A_down2 = UNetDown(64, 128)
        self.A_down3 = UNetDown(128, 256)
        self.A_down4 = UNetDown(256, 512, dropout=0.5)
        self.A_down5 = UNetDown(512, 512, dropout=0.5)
        self.A_down6 = UNetDown(512, 512, dropout=0.5)
        self.A_down7 = UNetDown(512, 512, dropout=0.5)
        self.A_down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.A_up1 = UNetUp(512, 512, dropout=0.5)
        self.A_up2 = UNetUp(1024, 512, dropout=0.5)
        self.A_up3 = UNetUp(1024, 512, dropout=0.5)
        self.A_up4 = UNetUp(1024, 512, dropout=0.5)
        self.A_up5 = UNetUp(1024, 256)
        self.A_up6 = UNetUp(512, 128)
        self.A_up7 = UNetUp(256, 64)

        self.A_final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 1, 4, padding=1),
            nn.Tanh(),
        )

        self.B_down1 = UNetDown(1, 64, normalize=False)
        self.B_down2 = UNetDown(64, 128)
        self.B_down3 = UNetDown(128, 256)
        self.B_down4 = UNetDown(256, 512, dropout=0.5)
        self.B_down5 = UNetDown(512, 512, dropout=0.5)
        self.B_down6 = UNetDown(512, 512, dropout=0.5)
        self.B_down7 = UNetDown(512, 512, dropout=0.5)
        self.B_down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.B_up1 = UNetUp(512, 512, dropout=0.5)
        self.B_up2 = UNetUp(1024, 512, dropout=0.5)
        self.B_up3 = UNetUp(1024, 512, dropout=0.5)
        self.B_up4 = UNetUp(1024, 512, dropout=0.5)
        self.B_up5 = UNetUp(1024, 256)
        self.B_up6 = UNetUp(512, 128)
        self.B_up7 = UNetUp(256, 64)

        self.B_final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 1, 4, padding=1),
            nn.Tanh(),
        )

        self.net2 = nn.Sequential(
            nn.Linear(256 * 256, 1601),
        )


    def forward(self, ha, vi):

        if ha.ndim != 2 or ha.shape[1] != 1601:
            raise ValueError(f"Expected tactile [B,1601], got {tuple(ha.shape)}")
        if vi.ndim != 4 or tuple(vi.shape[1:]) != (1, 256, 256) or vi.shape[0] != ha.shape[0]:
            raise ValueError(f"Expected images [B,1,256,256], got {tuple(vi.shape)}")
        # ha in
        h = self.net1(ha)
        h0 = h.reshape([h.shape[0]] + [1, 256, 256])
        h1 = self.A_down1(h0)
        h2 = self.A_down2(h1)
        h3 = self.A_down3(h2)
        h4 = self.A_down4(h3)
        h5 = self.A_down5(h4)
        h6 = self.A_down6(h5)
        h7 = self.A_down7(h6)
        h8 = self.A_down8(h7)
        ha_z = h8
        hz1 = self.A_up1(h8, h7)
        hz2 = self.A_up2(hz1, h6)
        hz3 = self.A_up3(hz2, h5)
        hz4 = self.A_up4(hz3, h4)
        hz5 = self.A_up5(hz4, h3)
        hz6 = self.A_up6(hz5, h2)
        hz7 = self.A_up7(hz6, h1)
        vi_output = self.A_final(hz7)

        # vi in
        v0 = vi
        v1 = self.B_down1(v0)
        v2 = self.B_down2(v1)
        v3 = self.B_down3(v2)
        v4 = self.B_down4(v3)
        v5 = self.B_down5(v4)
        v6 = self.B_down6(v5)
        v7 = self.B_down7(v6)
        v8 = self.B_down8(v7)
        vi_z = v8
        vz1 = self.B_up1(v8, v7)
        vz2 = self.B_up2(vz1, v6)
        vz3 = self.B_up3(vz2, v5)
        vz4 = self.B_up4(vz3, v4)
        vz5 = self.B_up5(vz4, v3)
        vz6 = self.B_up6(vz5, v2)
        vz7 = self.B_up7(vz6, v1)
        vz8 = self.B_final(vz7)
        vz9 = vz8.reshape([vz8.shape[0]] + [256*256])
        ha_output = self.net2(vz9)

        return ha_z, vi_z, ha_output, vi_output


if __name__ == "__main__":
    # ch = 32
    # source = torch.rand([4, ch, 64, 64])
    # model = DictResBottleneckBlock(ch)
    # y = model(source)
    # print(y.shape)
    source, target = torch.rand([4, 1601]), torch.rand([4, 1, 256, 256])
    model = Model()

    print(f'Total Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    x, y, x1, y1 = model(source, target)
    print(y.shape,x.shape, x1.shape,y1.shape)
