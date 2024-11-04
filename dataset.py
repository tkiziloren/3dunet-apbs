import os

import numpy as np
import torch
from Bio.PDB import PDBParser
from torch.utils.data import Dataset


class ProteinLigandDataset(Dataset):
    def __init__(self, root_dir, protein_names, transform=None, grid_size=161, voxel_size=1.0):
        """
        root_dir: Ana dizin yolu
        protein_names: Kullanılacak protein isimlerinin listesi
        transform: Veri artırımı ve standardizasyon işlemleri
        grid_size: 3D grid boyutu
        voxel_size: Her voxel'in boyutu (örneğin, 1.0 angstrom)
        """
        self.root_dir = root_dir
        self.protein_names = protein_names
        self.transform = transform
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for protein_name in self.protein_names:
            protein_path = os.path.join(self.root_dir, protein_name, f"{protein_name}_protein.pdb")
            pocket_path = os.path.join(self.root_dir, protein_name, f"{protein_name}_pocket.pdb")
            if os.path.exists(protein_path) and os.path.exists(pocket_path):
                samples.append((protein_path, pocket_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        protein_path, pocket_path = self.samples[idx]
        protein_grid = self.pdb_to_grid(protein_path)
        shape_info = self.get_shape_info(protein_grid)
        protein_input = torch.stack([protein_grid, shape_info], dim=0)

        pocket_label = self.pdb_to_grid(pocket_path)  # Pocket dosyasını grid olarak oku

        if self.transform:
            protein_input = self.transform(protein_input)
            pocket_label = self.transform(pocket_label)

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
            # Koordinatları gridin merkezine göre ölçeklendir ve kaydır
            grid_coord = np.floor(center + np.array(coord) / self.voxel_size).astype(int)

            # Koordinatların grid sınırları içinde olduğundan emin olun
            if (grid_coord >= 0).all() and (grid_coord < self.grid_size).all():
                grid[tuple(grid_coord)] = 1  # Voxeli 1 olarak işaretle

        return torch.tensor(grid, dtype=torch.uint8)  # PyTorch tensörü olarak döndür

    def get_shape_info(self, protein_grid):
        return torch.ones_like(protein_grid)
