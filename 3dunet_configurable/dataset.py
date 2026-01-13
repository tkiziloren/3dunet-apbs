import logging
import os
from typing import List
import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

# Logging yapılandırması
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path="config.yml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class ProteinLigandDatasetWithH5(Dataset):
    def __init__(self, h5_dir, protein_names, transform=None, config_path="config.yml"):
        self.h5_dir = h5_dir
        self.protein_names = protein_names
        self.transform = transform

        # Config’den feature ve label isimlerini çekiyoruz
        config = load_config(config_path)
        self.feature_names = config.get("features", ["electrostatic_grid", "shape"])
        self.label_name = config.get("label", "binding_site")
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples")

    def _load_samples(self):
        samples = []
        for protein_name in self.protein_names:
            h5_filepath = os.path.join(self.h5_dir, f"{protein_name}.h5")
            if not os.path.exists(h5_filepath):
                logging.error(f"H5 file for {protein_name} not found at {h5_filepath}")
                raise FileNotFoundError(f"Protein H5 file {h5_filepath} not found.")
            samples.append((protein_name, h5_filepath))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        protein_name, h5_filepath = self.samples[idx]

        with h5py.File(h5_filepath, "r") as h5f:
            # Feature tensor’u dinamik olarak oluştur
            features = []
            for feat in self.feature_names:
                # Hem /features/ altında hem de root’ta olabilir (senin yapına göre değişebilir)
                if feat in h5f["features"]:
                    arr = h5f["features"][feat][:]
                elif feat in h5f:
                    arr = h5f[feat][:]
                else:
                    raise KeyError(f"Feature '{feat}' not found in H5 file {h5_filepath}")
                features.append(arr)
            protein_input = torch.tensor(np.stack(features), dtype=torch.float32)

            # Label’ı dinamik oku
            if self.label_name in h5f.get("label", {}):
                pocket_label = torch.tensor(h5f["label"][self.label_name][:], dtype=torch.float32)
            elif self.label_name in h5f:
                pocket_label = torch.tensor(h5f[self.label_name][:], dtype=torch.float32)
            else:
                raise KeyError(f"Label '{self.label_name}' not found in H5 file {h5_filepath}")

        # Transform varsa uygula
        if self.transform:
            protein_input, pocket_label = self.transform(protein_input, pocket_label)

        return protein_input, pocket_label