import torch
import torch.nn as nn

class ConvNeXt3DBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)
        x = x + shortcut
        return x

class ConvNeXt3DV2(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_features=32, depths=[2,2,2,2]):
        super().__init__()
        # Encoder
        self.stem = nn.Conv3d(in_channels, base_features, kernel_size=4, stride=4)
        dims = [base_features, base_features*2, base_features*4, base_features*8]
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        in_dim = base_features
        for i in range(4):
            if i > 0:
                self.downsample_layers.append(nn.Conv3d(in_dim, dims[i], kernel_size=2, stride=2))
            else:
                self.downsample_layers.append(nn.Identity())
            blocks = [ConvNeXt3DBlock(dims[i]) for _ in range(depths[i])]
            self.stages.append(nn.Sequential(*blocks))
            in_dim = dims[i]

        # Decoder (U-Net skip connection)
        self.up3 = nn.ConvTranspose3d(dims[3], dims[2], kernel_size=2, stride=2)
        self.up_conv3 = nn.Conv3d(dims[2]*2, dims[2], kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose3d(dims[2], dims[1], kernel_size=2, stride=2)
        self.up_conv2 = nn.Conv3d(dims[1]*2, dims[1], kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose3d(dims[1], dims[0], kernel_size=2, stride=2)
        self.up_conv1 = nn.Conv3d(dims[0]*2, dims[0], kernel_size=3, padding=1)
        self.final_conv = nn.Conv3d(dims[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skips = []
        x = self.stem(x)
        for down, stage in zip(self.downsample_layers, self.stages):
            x = down(x)
            x = stage(x)
            skips.append(x)
        # skips: [enc1, enc2, enc3, enc4] - Sonuncusu bottleneck
        # Decoder
        x = skips[-1]
        x = self.up3(x)
        x = torch.cat([x, skips[2]], dim=1)
        x = self.up_conv3(x)

        x = self.up2(x)
        x = torch.cat([x, skips[1]], dim=1)
        x = self.up_conv2(x)

        x = self.up1(x)
        x = torch.cat([x, skips[0]], dim=1)
        x = self.up_conv1(x)

        x = self.final_conv(x)
        return x