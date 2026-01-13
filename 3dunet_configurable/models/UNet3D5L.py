import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super(SingleConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super(DoubleConv, self).__init__()
        self.conv1 = SingleConv(in_channels, out_channels, groups)
        self.conv2 = SingleConv(out_channels, out_channels, groups)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super(Encoder, self).__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv(in_channels, out_channels, groups)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, groups)

    def forward(self, x, skip):
        # Yukarı örnekleme
        x = self.upsample(x)

        # Yukarı örneklenen tensörün boyutları skip tensörünün boyutlarıyla uyuşmuyorsa boyutları eşleştir
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3) or x.size(4) != skip.size(4):
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)

        # Tensörleri birleştir
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet3D5L(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_features=32):
        super(UNet3D5L, self).__init__()
        self.enc1 = DoubleConv(in_channels, base_features)
        self.enc2 = Encoder(base_features, base_features * 2)
        self.enc3 = Encoder(base_features * 2, base_features * 4)
        self.enc4 = Encoder(base_features * 4, base_features * 8)
        self.enc5 = Encoder(base_features * 8, base_features * 16)
        self.bottleneck = DoubleConv(base_features * 16, base_features * 32)
        self.dec1 = Decoder(base_features * 32 + base_features * 16, base_features * 16)
        self.dec2 = Decoder(base_features * 16 + base_features * 8, base_features * 8)
        self.dec3 = Decoder(base_features * 8 + base_features * 4, base_features * 4)
        self.dec4 = Decoder(base_features * 4 + base_features * 2, base_features * 2)
        self.dec5 = Decoder(base_features * 2 + base_features, base_features)
        self.final_conv = nn.Conv3d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        bottleneck = self.bottleneck(enc5)
        dec1 = self.dec1(bottleneck, enc5)
        dec2 = self.dec2(dec1, enc4)
        dec3 = self.dec3(dec2, enc3)
        dec4 = self.dec4(dec3, enc2)
        dec5 = self.dec5(dec4, enc1)
        return self.final_conv(dec5)
