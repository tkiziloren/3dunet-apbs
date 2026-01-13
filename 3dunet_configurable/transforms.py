
import numpy as np
import torch
from scipy.ndimage import rotate

class MonaiWrapper:
    def __init__(self, monai_transform):
        self.monai_transform = monai_transform

    def __call__(self, protein, label):
        # MONAI transformlar çoğunlukla dict ya da tek tensora uygulanır.
        # Aynı augmentasyonu hem protein hem label'a uygula:
        data = {"image": protein, "label": label}
        out = self.monai_transform(data)
        return out["image"], out["label"]

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
            return protein, label  # Olasılığa göre dönüşüm uygulama

        # NumPy dizilerine dönüştür
        protein_np = protein.numpy()
        label_np = label.numpy()

        # Rastgele açılar ve eksenleri bir kez belirle
        angles = np.random.uniform(0, 360, size=3)
        axes = [(0, 1), (1, 2), (0, 2)]

        # Her iki nesneye de aynı açı ve eksenleri kullanarak dönüşüm uygula
        for axis, angle in zip(axes, angles):
            protein_np = rotate(protein_np, angle, axes=axis, reshape=False, order=3, mode='constant', cval=0)
            label_np = rotate(label_np, angle, axes=axis, reshape=False, order=0, mode='constant', cval=0)

        # Dönüştürülmüş dizileri geri Tensor'a çevir
        return torch.from_numpy(protein_np), torch.from_numpy(label_np)


class RandomFlip:
    """
    Rastgele eksenlerde tensor üzerinde yansıma (flip) işlemi uygular.
    """
    def __init__(self, axis_prob=0.5):
        self.axis_prob = axis_prob

    def __call__(self, protein, label):
        if torch.rand(1).item() < self.axis_prob:
            # Her bir tensorun kaç boyutlu olduğunu al
            p_dim = protein.dim()
            l_dim = label.dim()

            # Sadece uzaysal boyutları flip'le!
            # protein: (C, X, Y, Z) -> spatial: 1,2,3 | label: (X, Y, Z) -> spatial: 0,1,2
            spatial_axes_protein = list(range(p_dim - 3, p_dim))
            spatial_axes_label = list(range(l_dim))

            # Rastgele bir uzaysal eksen seç
            axis_p = np.random.choice(spatial_axes_protein)
            axis_l = np.random.choice(spatial_axes_label)  # İstersen axis_p - (p_dim-3) de diyebilirsin

            protein = torch.flip(protein, dims=[axis_p])
            label = torch.flip(label, dims=[axis_l])
        return protein, label


class Standardize:
    """
    Basit bir Z-skoru normalizasyonu uygular.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0, eps: float = 1e-5):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, protein, label):
        """
        Tensor'a Z-skoru normalizasyonu uygular.

        Args:
            tensor (torch.Tensor): Normalizasyon yapılacak giriş tensoru.

        Returns:
            torch.Tensor: Normalizasyon yapılmış tensor.
        """
        protein = (protein - protein.mean()) / (protein.std() + 1e-5)
        return protein, label
