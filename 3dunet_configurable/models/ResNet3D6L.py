import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class EncoderResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderResNet3D, self).__init__()
        self.pool = nn.MaxPool3d(2)
        self.block = BasicBlock3D(in_channels, out_channels)
    def forward(self, x):
        x = self.pool(x)
        return self.block(x)

class DecoderResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderResNet3D, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.block = BasicBlock3D(in_channels, out_channels)
    def forward(self, x, skip):
        x = self.upsample(x)
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3) or x.size(4) != skip.size(4):
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class ResNet3D6L(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_features=32):
        super(ResNet3D6L, self).__init__()
        self.enc1 = BasicBlock3D(in_channels, base_features)
        self.enc2 = EncoderResNet3D(base_features, base_features*2)
        self.enc3 = EncoderResNet3D(base_features*2, base_features*4)
        self.enc4 = EncoderResNet3D(base_features*4, base_features*8)
        self.enc5 = EncoderResNet3D(base_features*8, base_features*16)
        self.enc6 = EncoderResNet3D(base_features*16, base_features*32)
        self.bottleneck = BasicBlock3D(base_features*32, base_features*64)
        self.dec1 = DecoderResNet3D(base_features*64 + base_features*32, base_features*32)
        self.dec2 = DecoderResNet3D(base_features*32 + base_features*16, base_features*16)
        self.dec3 = DecoderResNet3D(base_features*16 + base_features*8, base_features*8)
        self.dec4 = DecoderResNet3D(base_features*8 + base_features*4, base_features*4)
        self.dec5 = DecoderResNet3D(base_features*4 + base_features*2, base_features*2)
        self.dec6 = DecoderResNet3D(base_features*2 + base_features, base_features)
        self.final_conv = nn.Conv3d(base_features, out_channels, kernel_size=1)
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        bottleneck = self.bottleneck(enc6)
        dec1 = self.dec1(bottleneck, enc6)
        dec2 = self.dec2(dec1, enc5)
        dec3 = self.dec3(dec2, enc4)
        dec4 = self.dec4(dec3, enc3)
        dec5 = self.dec5(dec4, enc2)
        dec6 = self.dec6(dec5, enc1)
        return self.final_conv(dec6)