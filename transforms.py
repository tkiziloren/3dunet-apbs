import random

import numpy as np
import torch
import torch.nn as nn


class RandomFlipAndRotate3D:
    def __call__(self, sample):
        # sample boyutları: (C, D, H, W) olarak varsayılır
        num_dims = sample.dim()

        # Eğer tensor boyutları (C, D, H, W) ise, uzaysal eksenler 1, 2 ve 3 olacaktır
        if num_dims == 4:
            if np.random.rand() > 0.5:
                # Flip için geçerli uzaysal eksenleri seçin
                sample = torch.flip(sample, [random.choice([1, 2, 3])])
            if np.random.rand() > 0.5:
                # Rotasyon için geçerli eksen çiftlerinden rastgele birini seçin
                dims = random.choice([[1, 2], [1, 3], [2, 3]])
                sample = torch.rot90(sample, k=random.randint(1, 3), dims=dims)
        return sample


class Standardize:
    def __call__(self, sample):
        # Veri tipini float32'ye dönüştür
        sample = sample.to(torch.float32)
        return (sample - sample.mean()) / (sample.std() + 1e-5)


class BCEDiceLoss(nn.Module):
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets)
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + 1) / (inputs.sum() + targets.sum() + 1)
        return bce_loss + dice_loss
