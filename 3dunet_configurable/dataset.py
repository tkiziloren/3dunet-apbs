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

# Feature normalization ranges
FEATURE_RANGES = {
    'electrostatic_grid': (-5.0, 5.0),  # kV
    'shape': (0.0, 1.0),  # Already normalized
    'hydrophobicity': (-4.5, 4.5),
    'dist_to_ligand': (0.0, 80.0),  # Å
    'dist_to_surface': (0.0, 80.0),  # Å
    # Atomic features are binary (0 or 1), no normalization needed
    'atomic_N': (0.0, 1.0),
    'atomic_O': (0.0, 1.0),
    'atomic_C': (0.0, 1.0),
    'atomic_P': (0.0, 1.0),
    'atomic_S': (0.0, 1.0),
    'atomic_donor': (0.0, 1.0),
    'atomic_acceptor': (0.0, 1.0),
    'atomic_hydrophobic': (0.0, 1.0),
    'atomic_aromatic': (0.0, 1.0),
    'atomic_halogen': (0.0, 1.0),
}

def normalize_feature(feature_array, feature_name):
    """
    Normalize feature to [0, 1] range based on known physical ranges
    """
    if feature_name not in FEATURE_RANGES:
        logging.warning(f"No normalization range defined for {feature_name}, returning as-is")
        return feature_array
    
    min_val, max_val = FEATURE_RANGES[feature_name]
    
    # Clip to range and normalize
    feature_clipped = np.clip(feature_array, min_val, max_val)
    feature_normalized = (feature_clipped - min_val) / (max_val - min_val)
    
    return feature_normalized


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
                
                # ✅ Normalize feature
                arr = normalize_feature(arr, feat)
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