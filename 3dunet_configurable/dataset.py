import logging
import math
import os
from typing import List
import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

# Logging yapılandırması
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Default feature normalization ranges.
FEATURE_RANGES = {
    'electrostatic_grid': (-5.0, 5.0),  # kV
    'shape': (0.0, 1.0),  # Already normalized
    'hydrophobicity': (-4.5, 4.5),
    'dist_to_ligand': (0.0, 80.0),  # Å
    'dist_to_surface': (0.0, 80.0),  # Å
    'electrostatic_positive_clip20': (0.0, 1.0),
    'electrostatic_negative_clip20': (0.0, 1.0),
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

def normalize_feature(feature_array, feature_name, normalization_overrides=None):
    """
    Normalize a feature using the default physical range or a config override.

    Override example:
      feature_normalization:
        electrostatic_grid:
          min: -10.0
          max: 10.0
          clip: true
          normalize: true

    Set clip=false and normalize=false to pass a feature through unchanged.
    """
    normalization_overrides = normalization_overrides or {}
    override = normalization_overrides.get(feature_name)
    if override is False:
        return feature_array

    if feature_name not in FEATURE_RANGES and override is None:
        logging.warning(f"No normalization range defined for {feature_name}, returning as-is")
        return feature_array

    params = {}
    if feature_name in FEATURE_RANGES:
        min_val, max_val = FEATURE_RANGES[feature_name]
        params.update({"min": min_val, "max": max_val, "clip": True, "normalize": True})
    if isinstance(override, dict):
        params.update(override)

    clip_enabled = bool(params.get("clip", True))
    normalize_enabled = bool(params.get("normalize", True))

    if "min" not in params or "max" not in params:
        if clip_enabled or normalize_enabled:
            raise ValueError(f"Feature normalization for {feature_name} requires min and max values")
        return feature_array

    min_val = float(params["min"])
    max_val = float(params["max"])
    if max_val <= min_val:
        raise ValueError(f"Invalid normalization range for {feature_name}: min={min_val}, max={max_val}")

    feature_processed = feature_array
    if clip_enabled:
        feature_processed = np.clip(feature_processed, min_val, max_val)
    if normalize_enabled:
        feature_processed = (feature_processed - min_val) / (max_val - min_val)
        output_min = params.get("output_min")
        output_max = params.get("output_max")
        if output_min is not None or output_max is not None:
            output_min = float(0.0 if output_min is None else output_min)
            output_max = float(1.0 if output_max is None else output_max)
            feature_processed = feature_processed * (output_max - output_min) + output_min

    return feature_processed


def load_config(config_path="config.yml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class ProteinLigandDatasetWithH5(Dataset):
    def __init__(self, h5_dir, protein_names, transform=None, config_path="config.yml"):
        self.h5_dir = h5_dir
        self.protein_names = protein_names
        self.transform = transform
        self._metadata_cache = {}

        # Config’den feature ve label isimlerini çekiyoruz
        config = load_config(config_path)
        self.feature_names = config.get("features", ["electrostatic_grid", "shape"])
        self.label_name = config.get("label", "binding_site")
        self.feature_normalization = config.get("feature_normalization", {})
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
                arr = self._read_feature_array(h5f, feat, h5_filepath)
                
                arr = normalize_feature(arr, feat, self.feature_normalization)
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

    def _read_feature_array(self, h5f, feat, h5_filepath):
        if feat == "electrostatic_positive_clip20":
            raw = self._read_feature_array(h5f, "electrostatic_grid", h5_filepath)
            return np.clip(raw, 0.0, 20.0).astype(np.float32) / 20.0
        if feat == "electrostatic_negative_clip20":
            raw = self._read_feature_array(h5f, "electrostatic_grid", h5_filepath)
            return np.clip(-raw, 0.0, 20.0).astype(np.float32) / 20.0

        if "features" in h5f and feat in h5f["features"]:
            return h5f["features"][feat][:]
        if feat in h5f:
            return h5f[feat][:]
        raise KeyError(f"Feature '{feat}' not found in H5 file {h5_filepath}")

    def get_metadata(self, idx):
        if idx in self._metadata_cache:
            return self._metadata_cache[idx]

        protein_name, h5_filepath = self.samples[idx]
        with h5py.File(h5_filepath, "r") as h5f:
            label_shape = self._read_label_shape(h5f)
            resolution = float(h5f.attrs.get("resolution", 1.0))
            box_size = int(h5f.attrs.get("box_size", label_shape[0]))
            physical_span = h5f.attrs.get(
                "physical_span_angstrom",
                h5f.attrs.get("physical_extent", resolution * max(box_size - 1, 1)),
            )
            physical_span = float(physical_span)
            max_distance = math.sqrt(3.0) * physical_span
            metadata = {
                "protein": protein_name,
                "h5_filepath": h5_filepath,
                "resolution": resolution,
                "box_size": box_size,
                "physical_span_angstrom": physical_span,
                "max_distance_angstrom": max_distance,
                "grid_origin": np.asarray(
                    h5f.attrs.get("grid_origin", h5f.attrs.get("electrostatic_grid_min", [0.0, 0.0, 0.0])),
                    dtype=np.float32,
                ),
            }

        self._metadata_cache[idx] = metadata
        return metadata

    def load_metric_mask(self, idx, group_name, dataset_name):
        _, h5_filepath = self.samples[idx]
        with h5py.File(h5_filepath, "r") as h5f:
            group = h5f.get(group_name)
            if group is not None and dataset_name in group:
                return group[dataset_name][:]
            auxiliary_group = h5f.get("auxiliary")
            if group_name == "features" and auxiliary_group is not None and dataset_name in auxiliary_group:
                return auxiliary_group[dataset_name][:]
            if dataset_name in h5f:
                return h5f[dataset_name][:]
        return None

    def _read_label_shape(self, h5f):
        label_group = h5f.get("label")
        if label_group is not None and self.label_name in label_group:
            return label_group[self.label_name].shape
        if self.label_name in h5f:
            return h5f[self.label_name].shape
        first_feature_name = self.feature_names[0]
        if "features" in h5f and first_feature_name in h5f["features"]:
            return h5f["features"][first_feature_name].shape
        if first_feature_name in h5f:
            return h5f[first_feature_name].shape
        raise KeyError(f"Cannot infer grid shape from H5 file: {h5f.filename}")
