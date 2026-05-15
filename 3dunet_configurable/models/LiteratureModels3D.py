import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(num_channels, preferred_groups=8):
    for groups in (preferred_groups, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


def _match_spatial_size(x, target_shape):
    if x.shape[2:] == target_shape:
        return x
    return F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)


class _ConvNormAct3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm="group", activation=True):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
        ]
        if norm == "batch":
            layers.append(nn.BatchNorm3d(out_channels))
        elif norm == "group":
            layers.append(nn.GroupNorm(_group_count(out_channels), out_channels))
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _PUResNetBottleneck3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.projection = None
        if stride != 1 or in_channels != out_channels:
            self.projection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        identity = x if self.projection is None else self.projection(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class _PUResNetUpBottleneck3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, target_shape):
        x = _match_spatial_size(x, target_shape)
        return self.relu(self.main(x) + self.shortcut(x))


class PUResNetV1Like3D(nn.Module):
    """Dense PyTorch version of the PUResNet v1 encoder-decoder topology."""

    def __init__(self, in_channels=18, out_channels=1, base_features=18, dropout=0.0):
        super().__init__()
        f = int(base_features)
        self.stage2a = _PUResNetBottleneck3D(in_channels, f, stride=1)
        self.stage2b = _PUResNetBottleneck3D(f, f, stride=1)
        self.stage2c = _PUResNetBottleneck3D(f, f, stride=1)

        self.stage4a = _PUResNetBottleneck3D(f, f * 2, stride=2)
        self.stage4b = _PUResNetBottleneck3D(f * 2, f * 2, stride=1)
        self.stage4f = _PUResNetBottleneck3D(f * 2, f * 2, stride=1)

        self.stage5a = _PUResNetBottleneck3D(f * 2, f * 4, stride=2)
        self.stage5b = _PUResNetBottleneck3D(f * 4, f * 4, stride=1)
        self.stage5c = _PUResNetBottleneck3D(f * 4, f * 4, stride=1)

        self.stage6a = _PUResNetBottleneck3D(f * 4, f * 8, stride=3)
        self.stage6b = _PUResNetBottleneck3D(f * 8, f * 8, stride=1)
        self.stage6c = _PUResNetBottleneck3D(f * 8, f * 8, stride=1)

        self.stage7a = _PUResNetBottleneck3D(f * 8, f * 16, stride=3)
        self.stage7b = _PUResNetBottleneck3D(f * 16, f * 16, stride=1)
        self.dropout = nn.Dropout3d(dropout) if dropout and dropout > 0 else nn.Identity()

        self.up8a = _PUResNetUpBottleneck3D(f * 16, f * 16)
        self.up8b = _PUResNetBottleneck3D(f * 16, f * 16, stride=1)
        self.up9a = _PUResNetUpBottleneck3D(f * 16 + f * 8, f * 8)
        self.up9b = _PUResNetBottleneck3D(f * 8, f * 8, stride=1)
        self.up10a = _PUResNetUpBottleneck3D(f * 8 + f * 4, f * 4)
        self.up10b = _PUResNetBottleneck3D(f * 4, f * 4, stride=1)
        self.up11a = _PUResNetUpBottleneck3D(f * 4 + f * 2, f * 2)
        self.up11b = _PUResNetBottleneck3D(f * 2, f * 2, stride=1)
        self.final_conv = nn.Conv3d(f * 2 + f, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.stage2a(x)
        x = self.stage2b(x)
        x1 = self.stage2c(x)

        x = self.stage4a(x1)
        x = self.stage4b(x)
        x2 = self.stage4f(x)

        x = self.stage5a(x2)
        x = self.stage5b(x)
        x3 = self.stage5c(x)

        x = self.stage6a(x3)
        x = self.stage6b(x)
        x4 = self.stage6c(x)

        x = self.stage7a(x4)
        x = self.dropout(self.stage7b(x))

        x = self.up8a(x, x4.shape[2:])
        x = self.up8b(x)
        x = torch.cat([x, x4], dim=1)
        x = self.up9a(x, x3.shape[2:])
        x = self.up9b(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up10a(x, x2.shape[2:])
        x = self.up10b(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up11a(x, x1.shape[2:])
        x = self.up11b(x)
        x = torch.cat([x, x1], dim=1)
        return self.final_conv(x)


class _KalasantyDoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class KalasantyUNet3D(nn.Module):
    """Kalasanty-style 3D U-Net with pooling schedule 2, 2, 3, 3."""

    def __init__(self, in_channels=18, out_channels=1, base_features=32, dropout=0.0):
        super().__init__()
        f = int(base_features)
        self.enc1 = _KalasantyDoubleConv3D(in_channels, f)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = _KalasantyDoubleConv3D(f, f * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = _KalasantyDoubleConv3D(f * 2, f * 4)
        self.pool3 = nn.MaxPool3d(3)
        self.enc4 = _KalasantyDoubleConv3D(f * 4, f * 8)
        self.pool4 = nn.MaxPool3d(3)
        self.bottleneck = _KalasantyDoubleConv3D(f * 8, f * 16)
        self.dropout = nn.Dropout3d(dropout) if dropout and dropout > 0 else nn.Identity()

        self.dec4 = _KalasantyDoubleConv3D(f * 16 + f * 8, f * 8)
        self.dec3 = _KalasantyDoubleConv3D(f * 8 + f * 4, f * 4)
        self.dec2 = _KalasantyDoubleConv3D(f * 4 + f * 2, f * 2)
        self.dec1 = _KalasantyDoubleConv3D(f * 2 + f, f)
        self.final_conv = nn.Conv3d(f, out_channels, kernel_size=1)

    def _decode(self, x, skip, block):
        x = _match_spatial_size(x, skip.shape[2:])
        return block(torch.cat([x, skip], dim=1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        x = self.dropout(self.bottleneck(self.pool4(e4)))

        x = self._decode(x, e4, self.dec4)
        x = self._decode(x, e3, self.dec3)
        x = self._decode(x, e2, self.dec2)
        x = self._decode(x, e1, self.dec1)
        return self.final_conv(x)


class _DenseV2ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            _ConvNormAct3D(channels, channels, norm="group"),
            _ConvNormAct3D(channels, channels, norm="group", activation=False),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class _DenseV2Stage3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, blocks=2):
        super().__init__()
        self.down = _ConvNormAct3D(in_channels, out_channels, stride=stride, norm="group")
        self.blocks = nn.Sequential(*(_DenseV2ResidualBlock3D(out_channels) for _ in range(blocks)))

    def forward(self, x):
        return self.blocks(self.down(x))


class _DenseV2Decoder3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, blocks=2):
        super().__init__()
        self.reduce = _ConvNormAct3D(in_channels + skip_channels, out_channels, kernel_size=1, norm="group")
        self.blocks = nn.Sequential(*(_DenseV2ResidualBlock3D(out_channels) for _ in range(blocks)))

    def forward(self, x, skip):
        x = _match_spatial_size(x, skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.blocks(self.reduce(x))


class PUResNetV2DenseLike3D(nn.Module):
    """Dense approximation of PUResNetV2's sparse residual encoder-decoder."""

    def __init__(self, in_channels=18, out_channels=1, base_features=24, dropout=0.1):
        super().__init__()
        f = int(base_features)
        self.enc1 = _DenseV2Stage3D(in_channels, f, stride=1, blocks=2)
        self.enc2 = _DenseV2Stage3D(f, f * 2, stride=2, blocks=2)
        self.enc3 = _DenseV2Stage3D(f * 2, f * 4, stride=2, blocks=3)
        self.enc4 = _DenseV2Stage3D(f * 4, f * 8, stride=2, blocks=3)
        self.bottleneck = _DenseV2Stage3D(f * 8, f * 16, stride=2, blocks=3)
        self.dropout = nn.Dropout3d(dropout) if dropout and dropout > 0 else nn.Identity()

        self.dec4 = _DenseV2Decoder3D(f * 16, f * 8, f * 8, blocks=2)
        self.dec3 = _DenseV2Decoder3D(f * 8, f * 4, f * 4, blocks=2)
        self.dec2 = _DenseV2Decoder3D(f * 4, f * 2, f * 2, blocks=2)
        self.dec1 = _DenseV2Decoder3D(f * 2, f, f, blocks=2)
        self.final_conv = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        x = self.dropout(self.bottleneck(e4))
        x = self.dec4(x, e4)
        x = self.dec3(x, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        return self.final_conv(x)


class _TransformerBottleneck3D(nn.Module):
    def __init__(self, channels, depth=2, num_heads=4, dropout=0.0):
        super().__init__()
        heads = min(num_heads, channels)
        while channels % heads != 0 and heads > 1:
            heads -= 1
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):
        batch, channels, depth, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        return tokens.transpose(1, 2).reshape(batch, channels, depth, height, width)


class SwinSiteLike3D(nn.Module):
    """Hybrid CNN + transformer U-Net candidate inspired by SwinSite."""

    def __init__(self, in_channels=18, out_channels=1, base_features=16, dropout=0.1):
        super().__init__()
        f = int(base_features)
        self.stem = _ConvNormAct3D(in_channels, f, norm="group")
        self.enc1 = _DenseV2ResidualBlock3D(f)
        self.down2 = _DenseV2Stage3D(f, f * 2, stride=2, blocks=1)
        self.down3 = _DenseV2Stage3D(f * 2, f * 4, stride=2, blocks=1)
        self.down4 = _DenseV2Stage3D(f * 4, f * 8, stride=2, blocks=1)
        self.down5 = _DenseV2Stage3D(f * 8, f * 16, stride=2, blocks=1)
        self.attention = _TransformerBottleneck3D(f * 16, depth=2, num_heads=4, dropout=dropout)
        self.dropout = nn.Dropout3d(dropout) if dropout and dropout > 0 else nn.Identity()

        self.dec4 = _DenseV2Decoder3D(f * 16, f * 8, f * 8, blocks=1)
        self.dec3 = _DenseV2Decoder3D(f * 8, f * 4, f * 4, blocks=1)
        self.dec2 = _DenseV2Decoder3D(f * 4, f * 2, f * 2, blocks=1)
        self.dec1 = _DenseV2Decoder3D(f * 2, f, f, blocks=1)
        self.final_conv = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(self.stem(x))
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)
        x = self.down5(e4)
        x = self.dropout(self.attention(x))
        x = self.dec4(x, e4)
        x = self.dec3(x, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        return self.final_conv(x)
