import torch

class MonaiWrapper:
    def __init__(self, monai_transform):
        self.monai_transform = monai_transform

    def __call__(self, protein, label):
        label_had_channel = label.dim() == 4
        if not label_had_channel:
            label = label.unsqueeze(0)

        data = {"image": protein, "label": label}
        out = self.monai_transform(data)
        out_label = out["label"] if label_had_channel else out["label"].squeeze(0)
        return out["image"], out_label

class CustomCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, protein, label):
        for transform in self.transforms:
            protein, label = transform(protein, label)
        return protein, label

class RandomRotate3D:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, protein, label):
        if torch.rand(1).item() > self.prob:
            return protein, label

        spatial_axis_pairs = [(0, 1), (1, 2), (0, 2)]
        for label_axes in spatial_axis_pairs:
            k = int(torch.randint(0, 4, (1,)).item())
            if k == 0:
                continue
            protein_axes = tuple(axis + (protein.dim() - 3) for axis in label_axes)
            protein = torch.rot90(protein, k=k, dims=protein_axes)
            label = torch.rot90(label, k=k, dims=label_axes)
        return protein.contiguous(), label.contiguous()


class RandomFlip:
    """
    Rastgele eksenlerde tensor üzerinde yansıma (flip) işlemi uygular.
    """
    def __init__(self, axis_prob=0.5):
        self.axis_prob = axis_prob

    def __call__(self, protein, label):
        for label_axis in range(3):
            if torch.rand(1).item() >= self.axis_prob:
                continue
            protein_axis = label_axis + (protein.dim() - 3)
            protein = torch.flip(protein, dims=[protein_axis])
            label = torch.flip(label, dims=[label_axis])
        return protein.contiguous(), label.contiguous()


class Standardize:
    """
    Basit bir Z-skoru normalizasyonu uygular.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0, eps: float = 1e-5, channel_wise: bool = True):
        self.mean = mean
        self.std = std
        self.eps = eps
        self.channel_wise = channel_wise

    def __call__(self, protein, label):
        """
        Tensor'a Z-skoru normalizasyonu uygular.

        Args:
            tensor (torch.Tensor): Normalizasyon yapılacak giriş tensoru.

        Returns:
            torch.Tensor: Normalizasyon yapılmış tensor.
        """
        if self.channel_wise and protein.dim() >= 4:
            spatial_dims = tuple(range(protein.dim() - 3, protein.dim()))
            mean = protein.mean(dim=spatial_dims, keepdim=True)
            std = protein.std(dim=spatial_dims, keepdim=True)
            std = torch.where(std < self.eps, torch.ones_like(std), std)
            protein = (protein - mean) / std
        else:
            protein = (protein - protein.mean()) / (protein.std() + self.eps)
        return protein, label
