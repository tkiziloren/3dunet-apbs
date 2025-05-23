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


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderAttention(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, groups=8):
        super(DecoderAttention, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.attn = AttentionBlock(in_channels, skip_channels, out_channels)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, groups)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
        skip = self.attn(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super(TripleConv, self).__init__()
        self.conv1 = SingleConv(in_channels, out_channels, groups)
        self.conv2 = SingleConv(out_channels, out_channels, groups)
        self.conv3 = SingleConv(out_channels, out_channels, groups)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


class UNet3D4LC(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_features=32):
        super(UNet3D4LC, self).__init__()
        self.enc1 = TripleConv(in_channels, base_features)
        self.enc2 = Encoder(base_features, base_features * 2)
        self.enc2.conv = TripleConv(base_features, base_features * 2)  # Encoder içindeki conv değiştirildi
        self.enc3 = Encoder(base_features * 2, base_features * 4)
        self.enc3.conv = TripleConv(base_features * 2, base_features * 4)
        self.enc4 = Encoder(base_features * 4, base_features * 8)
        self.enc4.conv = TripleConv(base_features * 4, base_features * 8)
        self.bottleneck = TripleConv(base_features * 8, base_features * 16)
        self.dec1 = Decoder(base_features * 16 + base_features * 8, base_features * 8)
        self.dec1.conv = TripleConv(base_features * 16, base_features * 8)
        self.dec2 = Decoder(base_features * 8 + base_features * 4, base_features * 4)
        self.dec2.conv = TripleConv(base_features * 8, base_features * 4)
        self.dec3 = Decoder(base_features * 4 + base_features * 2, base_features * 2)
        self.dec3.conv = TripleConv(base_features * 4, base_features * 2)
        self.dec4 = Decoder(base_features * 2 + base_features, base_features)
        self.dec4.conv = TripleConv(base_features * 2, base_features)
        self.final_conv = nn.Conv3d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        bottleneck = self.bottleneck(enc4)
        dec1 = self.dec1(bottleneck, enc4)
        dec2 = self.dec2(dec1, enc3)
        dec3 = self.dec3(dec2, enc2)
        dec4 = self.dec4(dec3, enc1)
        return self.final_conv(dec4)
