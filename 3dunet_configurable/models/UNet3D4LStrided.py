import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(num_channels, preferred_groups=8):
    for groups in (preferred_groups, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels, groups), out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv1 = SingleConv(in_channels, out_channels, groups)
        self.conv2 = SingleConv(out_channels, out_channels, groups)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class StridedEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels, groups), out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = DoubleConv(out_channels, out_channels, groups)

    def forward(self, x):
        return self.conv(self.downsample(x))


def _match_spatial_size(x, target_shape):
    slices = [slice(None), slice(None)]
    for axis, target_size in enumerate(target_shape, start=2):
        current_size = x.shape[axis]
        if current_size > target_size:
            start = (current_size - target_size) // 2
            slices.append(slice(start, start + target_size))
        else:
            slices.append(slice(None))
    x = x[tuple(slices)]

    pad_width = []
    for axis, target_size in reversed(list(enumerate(target_shape, start=2))):
        current_size = x.shape[axis]
        total_pad = max(target_size - current_size, 0)
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad
        pad_width.extend([left_pad, right_pad])
    if any(pad_width):
        x = F.pad(x, pad_width)
    return x


class RepeatUpsample3D(nn.Module):
    def forward(self, x):
        x = x.repeat_interleave(2, dim=2)
        x = x.repeat_interleave(2, dim=3)
        return x.repeat_interleave(2, dim=4)


class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, groups=8):
        super().__init__()
        self.upsample = RepeatUpsample3D()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, groups)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = _match_spatial_size(x, skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels, groups=8):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.GroupNorm(_group_count(inter_channels, groups), inter_channels),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.GroupNorm(_group_count(inter_channels, groups), inter_channels),
        )
        self.score = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, gate, skip):
        attention = self.score(self.gate_proj(gate) + self.skip_proj(skip))
        return skip * attention


class AttentionDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, groups=8):
        super().__init__()
        self.upsample = RepeatUpsample3D()
        self.attention = AttentionGate(in_channels, skip_channels, out_channels, groups)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, groups)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = _match_spatial_size(x, skip.shape[2:])
        skip = self.attention(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet3D4LStrided(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_features=32, dropout=0.5):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_features)
        self.enc2 = StridedEncoder(base_features, base_features * 2)
        self.enc3 = StridedEncoder(base_features * 2, base_features * 4)
        self.enc4 = StridedEncoder(base_features * 4, base_features * 8)
        self.bottleneck = StridedEncoder(base_features * 8, base_features * 16)
        self.dropout = nn.Dropout3d(p=dropout)

        self.dec1 = Decoder(base_features * 16, base_features * 8, base_features * 8)
        self.dec2 = Decoder(base_features * 8, base_features * 4, base_features * 4)
        self.dec3 = Decoder(base_features * 4, base_features * 2, base_features * 2)
        self.dec4 = Decoder(base_features * 2, base_features, base_features)
        self.final_conv = nn.Conv3d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        bottleneck = self.dropout(self.bottleneck(enc4))

        dec1 = self.dec1(bottleneck, enc4)
        dec2 = self.dec2(dec1, enc3)
        dec3 = self.dec3(dec2, enc2)
        dec4 = self.dec4(dec3, enc1)
        return self.final_conv(dec4)


class UNet3D4LAStrided(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_features=32, dropout=0.5):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_features)
        self.enc2 = StridedEncoder(base_features, base_features * 2)
        self.enc3 = StridedEncoder(base_features * 2, base_features * 4)
        self.enc4 = StridedEncoder(base_features * 4, base_features * 8)
        self.bottleneck = StridedEncoder(base_features * 8, base_features * 16)
        self.dropout = nn.Dropout3d(p=dropout)

        self.dec1 = AttentionDecoder(base_features * 16, base_features * 8, base_features * 8)
        self.dec2 = AttentionDecoder(base_features * 8, base_features * 4, base_features * 4)
        self.dec3 = AttentionDecoder(base_features * 4, base_features * 2, base_features * 2)
        self.dec4 = AttentionDecoder(base_features * 2, base_features, base_features)
        self.final_conv = nn.Conv3d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        bottleneck = self.dropout(self.bottleneck(enc4))

        dec1 = self.dec1(bottleneck, enc4)
        dec2 = self.dec2(dec1, enc3)
        dec3 = self.dec3(dec2, enc2)
        dec4 = self.dec4(dec3, enc1)
        return self.final_conv(dec4)
