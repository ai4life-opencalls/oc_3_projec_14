import torch
import torch.nn as nn
# import torch.nn.functional as F

from .base import BaseModel, DoubleConv


class Decoder(BaseModel):
    def __init__(self, num_classes: int = 1):
        """Uses all feature levels. Lowest feature resolution is 16x16"""
        super().__init__()
        self.name = "decoder"
        # input shape: B, 256, 16, 16
        self.conv_1 = DoubleConv(256, out_channels=1024, mid_channels=512)
        # in: B, 1024, 16, 16, out: B, 512, 32, 32
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)
        # concat with B, 256, 32, 32
        self.double_conv_1 = DoubleConv(768, 512)
        # in: B, 512, 32, 32, out: B, 256, 64, 64
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        # concat with B, 256, 64, 64
        self.double_conv_2 = DoubleConv(512, 256)
        # in: B, 512, 64, 64, out: B, 128, 128, 128
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        # concat with B, 256, 128, 128
        self.double_conv_3 = DoubleConv(384, 256)
        # in: B, 256, 128, 128, out: B, 128, 256, 256
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.double_conv_4 = DoubleConv(128, 128)
        # in: B, 256, 256, 256, out: B, 128, 512, 512
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        # in: B, 64, 512, 512
        self.double_conv_5 = DoubleConv(64, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(
        self, f3: torch.Tensor, f2: torch.Tensor, f1: torch.Tensor, f0: torch.Tensor
    ) -> torch.Tensor:
        out = self.conv_1(f3)
        out = self.deconv1(out)
        out = torch.cat((out, f2), dim=1)
        out = self.double_conv_1(out)
        out = self.deconv2(out)
        out = torch.cat((out, f1), dim=1)
        out = self.double_conv_2(out)
        out = self.deconv3(out)
        out = torch.cat((out, f0), dim=1)
        out = self.double_conv_3(out)
        out = self.deconv4(out)
        out = self.double_conv_4(out)
        out = self.deconv5(out)
        out = self.double_conv_5(out)
        out = self.final_conv(out)

        return out
