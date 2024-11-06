import torch
from torch import nn as nn


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """
    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


class BCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1.0):
        """
        alpha: Dice loss için ağırlıklandırma faktörü, 0 ile 1 arasında.
        smooth: Dice loss hesabında sıfır bölmeye karşı bir sabit ekleme.
        """
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Giriş ve hedef tensörlerin veri tiplerini float32'ye dönüştür
        inputs = inputs.to(torch.float32)
        targets = targets.to(torch.float32)

        # Binary Cross Entropy Loss hesapla
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets)

        # Sigmoid dönüşümü uygula
        inputs = torch.sigmoid(inputs)

        # Flatten the tensors
        inputs_flat = flatten(inputs)
        targets_flat = flatten(targets)

        # Dice Loss hesapla
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)

        # Toplam kaybı döndür (BCE + alpha * Dice)
        total_loss = bce_loss + self.alpha * dice_loss
        return total_loss



class BCEDiceLoss1(nn.Module):
    def forward(self, inputs, targets):
        # Giriş ve hedef tensörlerin veri tiplerini float32'ye dönüştür
        inputs = inputs.to(torch.float32)
        targets = targets.to(torch.float32)

        # Binary Cross Entropy Loss hesapla
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets)

        # Sigmoid dönüşümü uygula
        inputs = torch.sigmoid(inputs)

        # Dice Loss hesapla
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + 1) / (inputs.sum() + targets.sum() + 1)

        # Toplam kaybı döndür
        return bce_loss + dice_loss


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)
