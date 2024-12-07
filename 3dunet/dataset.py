import logging
import os
from typing import Tuple, List
import prody as pr
import h5py
import numpy as np
import torch
import yaml
from Bio.PDB import PDBParser
from torch.utils.data import Dataset
from potsim2 import PotGrid

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


class ProteinLigandDatasetWithH5(Dataset):
    def __init__(self, h5_dir, protein_names, transform=None, grid_size=161, voxel_size=1.0):
        self.h5_dir = h5_dir
        self.protein_names = protein_names
        self.transform = transform
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.samples = self._load_samples()
        print("Loaded {} samples".format(len(self.samples)))
        self.cache = {}  # Cache başlangıçta boş

        # Cache'i doldur ve eksik dosyaları oluştur
        for protein_name, protein_folder in self.samples:
            h5_filepath = os.path.join(protein_folder, protein_name + '_cache_grids.h5')

            if not os.path.exists(h5_filepath):
                # Eğer dosya yoksa oluştur
                self._create_cache(protein_name, protein_folder, h5_filepath)

            # Cache'i yükle
            with h5py.File(h5_filepath, 'r') as h5_file:
                self.cache[protein_name] = {
                    "electro_static_grid": h5_file["electro_static_grid"][:],
                    "pocket_label": h5_file["pocket_label"][:],
                    "shape_info": h5_file["shape_info"][:]
                }

        print("Loaded {} samples to the cache".format(len(self.samples)))

    def _load_samples(self):
        samples = []
        for protein_name in self.protein_names:
            protein_folder = os.path.join(self.h5_dir, protein_name)
            samples.append((protein_name, protein_folder))

            if not os.path.exists(protein_folder):
                logging.info(f"H5 file for {protein_name} not found.")
                raise FileNotFoundError(f"Protein directory {protein_folder} not found.")

            h5_filepath = os.path.join(protein_folder, protein_name + '_cache_grids.h5')

            # Eğer önbellek dosyası yoksa oluştur
            if not os.path.exists(h5_filepath):
                self._create_cache(protein_name, protein_folder, h5_filepath)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        protein_name, _ = self.samples[idx]  # Sadece protein adını alıyoruz çünkü cache'i kullanacağız

        # Cache'ten veriyi al
        cached_data = self.cache[protein_name]
        electro_static_grid = torch.tensor(cached_data["electro_static_grid"], dtype=torch.float32).clone().detach()
        pocket_label = torch.tensor(cached_data["pocket_label"], dtype=torch.float32).clone().detach()
        shape_info = torch.tensor(cached_data["shape_info"], dtype=torch.float32).clone().detach()

        # Eğer transform varsa uygula
        if self.transform:
            electro_static_grid, pocket_label = self.transform(electro_static_grid, pocket_label)

        # Model girişini oluştur
        protein_input = torch.stack([electro_static_grid, shape_info], dim=0)

        return protein_input, pocket_label

    def _create_cache(self, protein_name, protein_dir, h5_filepath):
        logging.info(f"Creating cache for {protein_name} at {h5_filepath}")

        # Protein ve PotGrid dosyalarını okuyun
        protein_file_path_prefix = os.path.join(protein_dir, protein_name)
        protein_file_path = protein_file_path_prefix + '.pdb'
        structure = pr.parsePDB(protein_file_path)

        potgrid = PotGrid(protein_file_path_prefix + "_selected.pdb", protein_file_path_prefix + "_grid.dx.gz")

        # Electrostatic grid ve pocket label'i okuyun
        grids_h5_path = os.path.join(protein_dir, protein_name + '_grids.h5')
        with h5py.File(grids_h5_path, 'r', swmr=True) as grids_h5:
            electro_static_grid = grids_h5["raw"][:]
            pocket_label = grids_h5["label"][:]

        # `shape_info` oluştur
        shape_info = self._get_shape_info(structure, potgrid)

        # H5 dosyasına kaydet
        with h5py.File(h5_filepath, 'w', swmr=True) as h5_file:
            h5_file.create_dataset("electro_static_grid", data=electro_static_grid, dtype=np.float32)
            h5_file.create_dataset("pocket_label", data=pocket_label, dtype=np.float32)
            h5_file.create_dataset("shape_info", data=shape_info, dtype=np.float32)

        logging.info(f"Cache created for {protein_name}.")

    def _get_shape_info(self, structure, grids):
        retgrid = np.zeros(shape=grids.grid.shape, dtype=np.float32)

        for i, coord in enumerate(structure.getCoords()):
            x, y, z = coord
            binx = int((x - min(grids.edges[0])) / grids.delta[0])
            biny = int((y - min(grids.edges[1])) / grids.delta[1])
            binz = int((z - min(grids.edges[2])) / grids.delta[2])

            if binx < grids.grid.shape[0] and biny < grids.grid.shape[1] and binz < grids.grid.shape[2]:
                retgrid[binx, biny, binz] = 1
        return retgrid


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
