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


class ConvNeXtBlock3D(nn.Module):
    def __init__(self, channels, expansion=4, layer_scale_init=1e-6):
        super().__init__()
        self.depthwise = nn.Conv3d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pointwise1 = nn.Linear(channels, expansion * channels)
        self.activation = nn.GELU()
        self.pointwise2 = nn.Linear(expansion * channels, channels)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(channels))

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.pointwise1(x)
        x = self.activation(x)
        x = self.pointwise2(x)
        x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)
        return x + residual


class ConvNeXtStage3D(nn.Module):
    def __init__(self, channels, depth=2):
        super().__init__()
        self.blocks = nn.Sequential(*(ConvNeXtBlock3D(channels) for _ in range(depth)))

    def forward(self, x):
        return self.blocks(x)


class ConvNeXtDecoder3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, depth=1):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )
        self.stage = ConvNeXtStage3D(out_channels, depth=depth)

    def forward(self, x, skip):
        x = _match_spatial_size(x, skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.stage(self.reduce(x))


class ConvNeXtUNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_features=32, dropout=0.2):
        super().__init__()
        features = [base_features, base_features * 2, base_features * 4, base_features * 8, base_features * 16]
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, features[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(features[0]), features[0]),
            nn.GELU(),
        )
        self.enc1 = ConvNeXtStage3D(features[0], depth=2)
        self.down2 = nn.Conv3d(features[0], features[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.enc2 = ConvNeXtStage3D(features[1], depth=2)
        self.down3 = nn.Conv3d(features[1], features[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.enc3 = ConvNeXtStage3D(features[2], depth=2)
        self.down4 = nn.Conv3d(features[2], features[3], kernel_size=3, stride=2, padding=1, bias=False)
        self.enc4 = ConvNeXtStage3D(features[3], depth=2)
        self.down5 = nn.Conv3d(features[3], features[4], kernel_size=3, stride=2, padding=1, bias=False)
        self.bottleneck = ConvNeXtStage3D(features[4], depth=2)
        self.dropout = nn.Dropout3d(dropout)

        self.dec4 = ConvNeXtDecoder3D(features[4], features[3], features[3], depth=1)
        self.dec3 = ConvNeXtDecoder3D(features[3], features[2], features[2], depth=1)
        self.dec2 = ConvNeXtDecoder3D(features[2], features[1], features[1], depth=1)
        self.dec1 = ConvNeXtDecoder3D(features[1], features[0], features[0], depth=1)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(self.stem(x))
        enc2 = self.enc2(self.down2(enc1))
        enc3 = self.enc3(self.down3(enc2))
        enc4 = self.enc4(self.down4(enc3))
        x = self.dropout(self.bottleneck(self.down5(enc4)))

        x = self.dec4(x, enc4)
        x = self.dec3(x, enc3)
        x = self.dec2(x, enc2)
        x = self.dec1(x, enc1)
        return self.final_conv(x)


class EfficientConvNeXtBlock3D(nn.Module):
    def __init__(self, channels, kernel_size=3, expansion=2, layer_scale_init=1e-4):
        super().__init__()
        padding = kernel_size // 2
        hidden = channels * expansion
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.pointwise = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
        )
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(1, channels, 1, 1, 1))

    def forward(self, x):
        return x + self.gamma * self.pointwise(self.norm(self.depthwise(x)))


class EfficientConvNeXtStage3D(nn.Module):
    def __init__(self, channels, depth=1, kernel_size=3, expansion=2):
        super().__init__()
        self.blocks = nn.Sequential(
            *(EfficientConvNeXtBlock3D(channels, kernel_size=kernel_size, expansion=expansion) for _ in range(depth))
        )

    def forward(self, x):
        return self.blocks(x)


class EfficientConvNeXtDecoder3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, depth=1, kernel_size=3, expansion=2):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )
        self.stage = EfficientConvNeXtStage3D(
            out_channels,
            depth=depth,
            kernel_size=kernel_size,
            expansion=expansion,
        )

    def forward(self, x, skip):
        x = _match_spatial_size(x, skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.stage(self.reduce(x))


class TinyConvNeXtUNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_features=32, dropout=0.1):
        super().__init__()
        features = [base_features, base_features * 2, base_features * 4, base_features * 8]
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, features[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(features[0]), features[0]),
            nn.GELU(),
        )
        self.enc1 = EfficientConvNeXtStage3D(features[0], depth=1, kernel_size=3, expansion=2)
        self.down2 = nn.Conv3d(features[0], features[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.enc2 = EfficientConvNeXtStage3D(features[1], depth=1, kernel_size=3, expansion=2)
        self.down3 = nn.Conv3d(features[1], features[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.enc3 = EfficientConvNeXtStage3D(features[2], depth=1, kernel_size=3, expansion=2)
        self.down4 = nn.Conv3d(features[2], features[3], kernel_size=3, stride=2, padding=1, bias=False)
        self.bottleneck = EfficientConvNeXtStage3D(features[3], depth=1, kernel_size=3, expansion=2)
        self.dropout = nn.Dropout3d(dropout)

        self.dec3 = EfficientConvNeXtDecoder3D(features[3], features[2], features[2], depth=1)
        self.dec2 = EfficientConvNeXtDecoder3D(features[2], features[1], features[1], depth=1)
        self.dec1 = EfficientConvNeXtDecoder3D(features[1], features[0], features[0], depth=1)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(self.stem(x))
        enc2 = self.enc2(self.down2(enc1))
        enc3 = self.enc3(self.down3(enc2))
        x = self.dropout(self.bottleneck(self.down4(enc3)))
        x = self.dec3(x, enc3)
        x = self.dec2(x, enc2)
        x = self.dec1(x, enc1)
        return self.final_conv(x)


class SqueezeExcite3D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=4, spatial_kernel_size=7):
        super().__init__()
        hidden = max(channels // reduction, 1)
        padding = spatial_kernel_size // 2
        self.channel_mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=spatial_kernel_size, padding=padding, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_gate = self.channel_mlp(F.adaptive_avg_pool3d(x, 1))
        max_gate = self.channel_mlp(F.adaptive_max_pool3d(x, 1))
        x = x * torch.sigmoid(avg_gate + max_gate)

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.spatial_gate(torch.cat([avg_map, max_map], dim=1))


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=False, use_cbam=False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.GELU()
        if use_cbam:
            self.attention = CBAM3D(out_channels)
        elif use_se:
            self.attention = SqueezeExcite3D(out_channels)
        else:
            self.attention = nn.Identity()
        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.attention(x)
        return self.activation(x + residual)


class ResidualEncoder3D(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=False, use_cbam=False):
        super().__init__()
        self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.block = ResidualBlock3D(out_channels, out_channels, use_se=use_se, use_cbam=use_cbam)

    def forward(self, x):
        return self.block(self.downsample(x))


class ResidualDecoder3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_se=False, use_cbam=False):
        super().__init__()
        self.block = ResidualBlock3D(in_channels + skip_channels, out_channels, use_se=use_se, use_cbam=use_cbam)

    def forward(self, x, skip):
        x = _match_spatial_size(x, skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class _ResidualUNet3DBase(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        base_features=32,
        dropout=0.2,
        use_se=False,
        use_cbam=False,
    ):
        super().__init__()
        features = [base_features, base_features * 2, base_features * 4, base_features * 8, base_features * 16]
        self.enc1 = ResidualBlock3D(in_channels, features[0], use_se=use_se, use_cbam=use_cbam)
        self.enc2 = ResidualEncoder3D(features[0], features[1], use_se=use_se, use_cbam=use_cbam)
        self.enc3 = ResidualEncoder3D(features[1], features[2], use_se=use_se, use_cbam=use_cbam)
        self.enc4 = ResidualEncoder3D(features[2], features[3], use_se=use_se, use_cbam=use_cbam)
        self.bottleneck = ResidualEncoder3D(features[3], features[4], use_se=use_se, use_cbam=use_cbam)
        self.dropout = nn.Dropout3d(dropout)

        self.dec4 = ResidualDecoder3D(features[4], features[3], features[3], use_se=use_se, use_cbam=use_cbam)
        self.dec3 = ResidualDecoder3D(features[3], features[2], features[2], use_se=use_se, use_cbam=use_cbam)
        self.dec2 = ResidualDecoder3D(features[2], features[1], features[1], use_se=use_se, use_cbam=use_cbam)
        self.dec1 = ResidualDecoder3D(features[1], features[0], features[0], use_se=use_se, use_cbam=use_cbam)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        x = self.dropout(self.bottleneck(enc4))

        x = self.dec4(x, enc4)
        x = self.dec3(x, enc3)
        x = self.dec2(x, enc2)
        x = self.dec1(x, enc1)
        return self.final_conv(x)


class ResidualUNet3D(_ResidualUNet3DBase):
    def __init__(self, in_channels=2, out_channels=1, base_features=32, dropout=0.2):
        super().__init__(in_channels, out_channels, base_features, dropout=dropout, use_se=False)


class SEResUNet3D(_ResidualUNet3DBase):
    def __init__(self, in_channels=2, out_channels=1, base_features=32, dropout=0.2):
        super().__init__(in_channels, out_channels, base_features, dropout=dropout, use_se=True)


class CBAMUNet3D(_ResidualUNet3DBase):
    def __init__(self, in_channels=2, out_channels=1, base_features=32, dropout=0.2):
        super().__init__(in_channels, out_channels, base_features, dropout=dropout, use_cbam=True)


class ResNetGNBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=False, use_cbam=False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.ReLU(inplace=True)
        if use_cbam:
            self.attention = CBAM3D(out_channels)
        elif use_se:
            self.attention = SqueezeExcite3D(out_channels)
        else:
            self.attention = nn.Identity()
        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.attention(x)
        return self.activation(x + residual)


class ResNetGNEncoder3D(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=False, use_cbam=False):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.block = ResNetGNBlock3D(in_channels, out_channels, use_se=use_se, use_cbam=use_cbam)

    def forward(self, x):
        return self.block(self.pool(x))


class ResNetGNDecoder3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_se=False, use_cbam=False):
        super().__init__()
        self.block = ResNetGNBlock3D(in_channels + skip_channels, out_channels, use_se=use_se, use_cbam=use_cbam)

    def forward(self, x, skip):
        x = _match_spatial_size(x, skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class _ResNet3D4LGNBase(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        base_features=32,
        dropout=0.1,
        use_se=False,
        use_cbam=False,
    ):
        super().__init__()
        features = [base_features, base_features * 2, base_features * 4, base_features * 8, base_features * 16]
        self.enc1 = ResNetGNBlock3D(in_channels, features[0], use_se=use_se, use_cbam=use_cbam)
        self.enc2 = ResNetGNEncoder3D(features[0], features[1], use_se=use_se, use_cbam=use_cbam)
        self.enc3 = ResNetGNEncoder3D(features[1], features[2], use_se=use_se, use_cbam=use_cbam)
        self.enc4 = ResNetGNEncoder3D(features[2], features[3], use_se=use_se, use_cbam=use_cbam)
        self.bottleneck = ResNetGNBlock3D(features[3], features[4], use_se=use_se, use_cbam=use_cbam)
        self.dropout = nn.Dropout3d(dropout)

        self.dec1 = ResNetGNDecoder3D(features[4], features[3], features[3], use_se=use_se, use_cbam=use_cbam)
        self.dec2 = ResNetGNDecoder3D(features[3], features[2], features[2], use_se=use_se, use_cbam=use_cbam)
        self.dec3 = ResNetGNDecoder3D(features[2], features[1], features[1], use_se=use_se, use_cbam=use_cbam)
        self.dec4 = ResNetGNDecoder3D(features[1], features[0], features[0], use_se=use_se, use_cbam=use_cbam)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        x = self.dropout(self.bottleneck(enc4))
        x = self.dec1(x, enc4)
        x = self.dec2(x, enc3)
        x = self.dec3(x, enc2)
        x = self.dec4(x, enc1)
        return self.final_conv(x)


class ResNet3D4LGN(_ResNet3D4LGNBase):
    def __init__(self, in_channels=2, out_channels=1, base_features=32, dropout=0.1):
        super().__init__(in_channels, out_channels, base_features, dropout=dropout)


class SEResNet3D4LGN(_ResNet3D4LGNBase):
    def __init__(self, in_channels=2, out_channels=1, base_features=32, dropout=0.1):
        super().__init__(in_channels, out_channels, base_features, dropout=dropout, use_se=True)


class CBAMResNet3D4LGN(_ResNet3D4LGNBase):
    def __init__(self, in_channels=2, out_channels=1, base_features=32, dropout=0.1):
        super().__init__(in_channels, out_channels, base_features, dropout=dropout, use_cbam=True)


class PlainConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class LightweightUNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_features=32, dropout=0.1):
        super().__init__()
        features = [base_features, base_features * 2, base_features * 4, base_features * 8]
        self.enc1 = PlainConvBlock3D(in_channels, features[0])
        self.enc2 = ResidualEncoder3D(features[0], features[1])
        self.enc3 = ResidualEncoder3D(features[1], features[2])
        self.bottleneck = ResidualEncoder3D(features[2], features[3])
        self.dropout = nn.Dropout3d(dropout)

        self.dec3 = ResidualDecoder3D(features[3], features[2], features[2])
        self.dec2 = ResidualDecoder3D(features[2], features[1], features[1])
        self.dec1 = ResidualDecoder3D(features[1], features[0], features[0])
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        x = self.dropout(self.bottleneck(enc3))
        x = self.dec3(x, enc3)
        x = self.dec2(x, enc2)
        x = self.dec1(x, enc1)
        return self.final_conv(x)


class UNetPlusPlus3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_features=32, dropout=0.2):
        super().__init__()
        features = [base_features, base_features * 2, base_features * 4, base_features * 8, base_features * 16]
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(dropout)

        self.conv0_0 = PlainConvBlock3D(in_channels, features[0])
        self.conv1_0 = PlainConvBlock3D(features[0], features[1])
        self.conv2_0 = PlainConvBlock3D(features[1], features[2])
        self.conv3_0 = PlainConvBlock3D(features[2], features[3])
        self.conv4_0 = PlainConvBlock3D(features[3], features[4])

        self.conv0_1 = PlainConvBlock3D(features[0] + features[1], features[0])
        self.conv1_1 = PlainConvBlock3D(features[1] + features[2], features[1])
        self.conv2_1 = PlainConvBlock3D(features[2] + features[3], features[2])
        self.conv3_1 = PlainConvBlock3D(features[3] + features[4], features[3])

        self.conv0_2 = PlainConvBlock3D(features[0] * 2 + features[1], features[0])
        self.conv1_2 = PlainConvBlock3D(features[1] * 2 + features[2], features[1])
        self.conv2_2 = PlainConvBlock3D(features[2] * 2 + features[3], features[2])

        self.conv0_3 = PlainConvBlock3D(features[0] * 3 + features[1], features[0])
        self.conv1_3 = PlainConvBlock3D(features[1] * 3 + features[2], features[1])

        self.conv0_4 = PlainConvBlock3D(features[0] * 4 + features[1], features[0])
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def _up(self, x, skip):
        return _match_spatial_size(x, skip.shape[2:])

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.dropout(self.conv4_0(self.pool(x3_0)))

        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], dim=1))
        return self.final_conv(x0_4)
