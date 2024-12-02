import logging
import os
from typing import Tuple, List

import h5py
import numpy as np
import torch
import yaml
from Bio.PDB import PDBParser
from torch.utils.data import Dataset

# Logging yapılandırması
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

VDW_RADII = {
    'H': 1.2,  # Hidrojen
    'C': 1.7,  # Karbon
    'N': 1.55,  # Azot
    'O': 1.52,  # Oksijen
    'S': 1.8,  # Kükürt
    'P': 1.8,  # Fosfor
    'Cl': 1.75,  # Klor
    'F': 1.47,  # Flor
    'Mg': 1.73,  # Magnezyum
    'Ca': 2.31,  # Kalsiyum
    'Fe': 2.0,  # Demir
    'Zn': 1.39,  # Çinko
    'Cu': 1.4,  # Bakır
    'Mn': 1.8,  # Mangan
    'Co': 2.0,  # Kobalt
    'Br': 1.85,  # Brom
    'I': 1.98,  # İyot
    'Se': 1.9  # Selenyum
}


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class ProteinLigandDataset(Dataset):
    def __init__(self, root_dir, cache_dir, protein_names, transform=None, grid_size=161, voxel_size=1.0):
        """
        root_dir: Ana dizin yolu
        protein_names: Kullanılacak protein isimlerinin listesi
        transform: Veri artırımı ve standardizasyon işlemleri
        grid_size: 3D grid boyutu
        voxel_size: Her voxel'in boyutu
        """
        self.root_dir = root_dir
        self.protein_names = protein_names
        self.transform = transform
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)  # cache dizinini oluştur
        self.samples = self._load_samples()

    def _load_samples(self):
        # Tüm protein isimleri için H5 dosyalarının yolunu al
        samples = []
        for protein_name in self.protein_names:
            h5_filepath = os.path.join(self.cache_dir, f"{protein_name}.h5")
            samples.append((protein_name, h5_filepath))
        return samples

    def __len__(self):
        return len(self.samples)

    def read_h5_file(self, h5_filepath):
        with h5py.File(h5_filepath, 'r') as h5_file:
            protein_grid = h5_file["protein"][:]
            pocket_label = h5_file["pocket"][:]
            coords_array = h5_file["coords"][:]
            elements_array = h5_file["elements"][:]

        # Veri doğrulama
        assert protein_grid.shape == (161, 161, 161), f"Unexpected protein grid shape: {protein_grid.shape}"
        assert pocket_label.shape == (161, 161, 161), f"Unexpected pocket label shape: {pocket_label.shape}"
        assert coords_array.shape[1] == 3, f"Unexpected coords shape: {coords_array.shape}"

        # Coord ve elementleri birleştir
        coords = [
            (coord, element.decode('utf-8') if isinstance(element, bytes) else element)
            for coord, element in zip(coords_array, elements_array)
        ]

        # Tensor'lara dönüştür
        protein_grid = torch.tensor(protein_grid, dtype=torch.float32)
        pocket_label = torch.tensor(pocket_label, dtype=torch.float32)
        return protein_grid, pocket_label, coords

    def __getitem__(self, idx):
        protein_name, h5_filepath = self.samples[idx]
        if not os.path.exists(h5_filepath):
            logging.info(f"H5 file for {protein_name} not found. Creating new file.")
            protein_path = os.path.join(self.root_dir, protein_name, f"{protein_name}_protein.pdb")
            pocket_path = os.path.join(self.root_dir, protein_name, f"{protein_name}_pocket.pdb")

            protein_grid, coords = self.pdb_to_grid(protein_path)
            pocket_label, _ = self.pdb_to_grid(pocket_path)

            coords_array = np.array([coord.tolist() for coord, _ in coords], dtype=np.float32)
            elements_array = np.array([element for _, element in coords], dtype=h5py.string_dtype(encoding='utf-8'))

            with h5py.File(h5_filepath, 'w') as h5_file:
                h5_file.create_dataset("protein", data=protein_grid.astype(np.float32), dtype=np.float32)
                h5_file.create_dataset("pocket", data=pocket_label.astype(np.float32), dtype=np.float32)
                h5_file.create_dataset("coords", data=coords_array, dtype=np.float32)
                h5_file.create_dataset("elements", data=elements_array)
        else:
            protein_grid, pocket_label, coords = self.read_h5_file(h5_filepath)

        shape_info = self.generate_shape_info(protein_grid, coords)

        if self.transform:
            protein_grid, pocket_label = self.transform(protein_grid, pocket_label)

        protein_input = torch.stack([protein_grid, shape_info], dim=0)

        return protein_input, pocket_label

    def pdb_to_grid(self, pdb_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('', pdb_path)
        coords = [(np.array(atom.get_coord()), atom.element) for atom in structure.get_atoms() if atom.element != 'H']

        grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.uint8)
        center = np.array([self.grid_size // 2] * 3)

        for coord, element in coords:
            grid_coord = np.floor(center + np.array(coord) / self.voxel_size).astype(int)
            if (grid_coord >= 0).all() and (grid_coord < self.grid_size).all():
                grid[tuple(grid_coord)] = 1

        return grid, coords

    def generate_shape_info(self, protein_grid, coords: List[Tuple[np.ndarray, str]]):
        if isinstance(protein_grid, torch.Tensor):
            protein_grid = protein_grid.numpy()

        shape_info = np.zeros_like(protein_grid, dtype=np.float32)
        center = np.array([self.grid_size // 2] * 3)

        for coord, element in coords:
            radius = VDW_RADII.get(element, 1.5)
            radius_in_voxels = int(np.ceil(radius / self.voxel_size))

            x, y, z = np.ogrid[-radius_in_voxels:radius_in_voxels + 1,
                      -radius_in_voxels:radius_in_voxels + 1,
                      -radius_in_voxels:radius_in_voxels + 1]
            mask = x ** 2 + y ** 2 + z ** 2 <= radius_in_voxels ** 2

            grid_coord = np.floor(center + np.array(coord) / self.voxel_size).astype(int)

            for dx, dy, dz in zip(*np.where(mask)):
                neighbor = grid_coord + np.array([dx - radius_in_voxels,
                                                  dy - radius_in_voxels,
                                                  dz - radius_in_voxels])
                if (neighbor >= 0).all() and (neighbor < self.grid_size).all():
                    shape_info[tuple(neighbor)] = 1.0

        return torch.tensor(shape_info, dtype=torch.float32)
