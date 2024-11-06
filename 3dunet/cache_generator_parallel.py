import os
import h5py
import yaml
import logging
import numpy as np
from Bio.PDB import PDBParser
from concurrent.futures import ProcessPoolExecutor, as_completed

# Logging yapılandırması
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Global ayarlar
GRID_SIZE = 161
VOXEL_SIZE = 1.0
CACHE_DIR = 'cache'


# Dizini oluştur



def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def pdb_to_grid(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('', pdb_path)
    coords = [atom.get_coord() for atom in structure.get_atoms() if atom.element != 'H']

    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    center = np.array([GRID_SIZE // 2] * 3)

    for coord in coords:
        grid_coord = np.floor(center + np.array(coord) / VOXEL_SIZE).astype(int)
        if (grid_coord >= 0).all() and (grid_coord < GRID_SIZE).all():
            grid[tuple(grid_coord)] = 1

    return grid


def generate_h5_file(protein_name, root_dir, cache_dir):
    h5_filepath = os.path.join(cache_dir, f"{protein_name}.h5")

    if os.path.exists(h5_filepath):
        logging.info(f"H5 file for {protein_name} already exists. Skipping.")
        return  # Dosya zaten varsa atla

    protein_path = os.path.join(root_dir, protein_name, f"{protein_name}_protein.pdb")
    pocket_path = os.path.join(root_dir, protein_name, f"{protein_name}_pocket.pdb")

    if not os.path.exists(protein_path) or not os.path.exists(pocket_path):
        logging.warning(f"Protein or pocket file for {protein_name} not found. Skipping.")
        return

    protein_grid = pdb_to_grid(protein_path).astype(np.float32)
    pocket_label = pdb_to_grid(pocket_path).astype(np.float32)

    # H5 dosyasını oluştur ve veriyi kaydet
    with h5py.File(h5_filepath, 'w') as h5_file:
        h5_file.create_dataset("protein", data=protein_grid, dtype=np.float32)
        h5_file.create_dataset("pocket", data=pocket_label, dtype=np.float32)

    logging.info(f"H5 file created for {protein_name}")


def generate_cache_for_proteins(protein_names, root_dir, cache_dir, max_workers=10):
    # Çoklu işlem havuzunu başlat
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_h5_file, protein_name, root_dir, cache_dir): protein_name for protein_name in protein_names}

        for future in as_completed(futures):
            protein_name = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing {protein_name}: {e}")


if __name__ == "__main__":
    # Config dosyasından protein isimlerini yükle
    config = load_config("./config/config.yml")
    data_dir = config["data_directory"]
    cache_dir = config["cache_directory"]
    os.makedirs(cache_dir, exist_ok=True)
    logging.info(f"Data directory for {config['data_directory']}")
    logging.info(f"Cache directory for {config['data_directory']}")

    # Tüm protein isimlerini train, validation ve test setlerinden birleştir
    protein_names = config["datasets"]["train"] + config["datasets"]["validation"] + config["datasets"]["test"]

    # Cache dosyalarını oluştur
    generate_cache_for_proteins(protein_names, data_dir, cache_dir)