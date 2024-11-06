import logging
import os

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

    def __getitem__(self, idx):
        protein_name, h5_filepath = self.samples[idx]

        # Eğer h5 dosyası yoksa dosyayı oluştur ve protein ile pocket verilerini kaydet
        if not os.path.exists(h5_filepath):
            logging.info(f"H5 file for {protein_name} not found. Creating new file.")
            protein_path = os.path.join(self.root_dir, protein_name, f"{protein_name}_protein.pdb")
            pocket_path = os.path.join(self.root_dir, protein_name, f"{protein_name}_pocket.pdb")

            protein_grid = self.pdb_to_grid(protein_path).astype(np.float32)
            pocket_label = self.pdb_to_grid(pocket_path).astype(np.float32)

            # H5 dosyasını oluştur ve veriyi kaydet
            with h5py.File(h5_filepath, 'w') as h5_file:
                h5_file.create_dataset("protein", data=protein_grid)
                h5_file.create_dataset("pocket", data=pocket_label)
        else:
            # H5 dosyasını açarak protein ve pocket grid verilerini oku
            with h5py.File(h5_filepath, "r") as h5_file:
                protein_grid = torch.tensor(h5_file["protein"][:], dtype=torch.float32)
                pocket_label = torch.tensor(h5_file["pocket"][:], dtype=torch.float32)

        # Shape bilgisi al
        shape_info = self.generate_shape_info(protein_grid)

        # Dönüşüm varsa, protein grid'i ve pocket label'a aynı anda uygula
        if self.transform:
            protein_grid, pocket_label = self.transform(protein_grid, pocket_label)

        # Protein input'u oluştur (protein grid ve shape bilgisi)
        protein_input = torch.stack([protein_grid, shape_info], dim=0)

        return protein_input, pocket_label

    def pdb_to_grid(self, pdb_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('', pdb_path)
        coords = [atom.get_coord() for atom in structure.get_atoms() if atom.element != 'H']

        # 3D grid oluştur ve başlangıçta tüm voxelleri 0 olarak ayarla
        grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.uint8)
        center = np.array([self.grid_size // 2] * 3)

        # Atom koordinatlarını grid üzerine yerleştir
        for coord in coords:
            grid_coord = np.floor(center + np.array(coord) / self.voxel_size).astype(int)
            if (grid_coord >= 0).all() and (grid_coord < self.grid_size).all():
                grid[tuple(grid_coord)] = 1

        return grid

    def generate_shape_info(protein_grid, coords, grid_size=161, voxel_size=1.0):
        """
        Proteinin atom büyüklüğüne dayalı iskelet yapısını çıkarır.

        Args:
            protein_grid (np.ndarray): Proteinin yoğunluk haritasını temsil eden 3D grid.
            coords (list): Her bir atomun koordinatı ve elementi (atom büyüklüğü).

        Returns:
            shape_info (np.ndarray): İskelet yapıyı temsil eden 3D grid.
        """
        shape_info = np.zeros_like(protein_grid, dtype=np.float32)
        center = np.array([grid_size // 2] * 3)

        for coord, element in coords:
            # Atomun vdW yarıçapını belirle
            radius = VDW_RADII.get(element, 1.5)  # Bilinmeyen elementler için varsayılan yarıçap
            radius_in_voxels = int(np.ceil(radius / voxel_size))

            # Atomun merkezini grid üzerinde hesapla
            grid_coord = np.floor(center + np.array(coord) / voxel_size).astype(int)

            # Atom etrafında vdW yarıçapına göre voxelleri işaretle
            for x in range(-radius_in_voxels, radius_in_voxels + 1):
                for y in range(-radius_in_voxels, radius_in_voxels + 1):
                    for z in range(-radius_in_voxels, radius_in_voxels + 1):
                        if np.linalg.norm([x, y, z]) <= radius_in_voxels:
                            neighbor = grid_coord + np.array([x, y, z])
                            if (neighbor >= 0).all() and (neighbor < grid_size).all():
                                shape_info[tuple(neighbor)] = 1.0
