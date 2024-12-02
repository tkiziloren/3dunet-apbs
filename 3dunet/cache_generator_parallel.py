import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import yaml
from Bio.PDB import PDBParser

# Logging yapılandırması
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

GRID_SIZE = 161
VOXEL_SIZE = 1.0


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def pdb_to_grid(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('', pdb_path)
    coords = [(atom.get_coord(), atom.element) for atom in structure.get_atoms() if atom.element != 'H']

    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    center = np.array([GRID_SIZE // 2] * 3)

    for coord, element in coords:
        grid_coord = np.floor(center + np.array(coord) / VOXEL_SIZE).astype(int)
        if (grid_coord >= 0).all() and (grid_coord < GRID_SIZE).all():
            grid[tuple(grid_coord)] = 1

    return grid, coords


def generate_h5_file(protein_name, root_dir, cache_dir):
    h5_filepath = os.path.join(cache_dir, f"{protein_name}.h5")

    if os.path.exists(h5_filepath):
        logging.info(f"H5 file for {protein_name} already exists. Skipping.")
        return

    protein_path = os.path.join(root_dir, protein_name, f"{protein_name}_protein.pdb")
    pocket_path = os.path.join(root_dir, protein_name, f"{protein_name}_pocket.pdb")

    if not os.path.exists(protein_path) or not os.path.exists(pocket_path):
        logging.warning(f"Protein or pocket file for {protein_name} not found. Skipping.")
        return

    protein_grid, coords = pdb_to_grid(protein_path)
    pocket_label, _ = pdb_to_grid(pocket_path)

    coords_array = np.array([coord.tolist() for coord, _ in coords], dtype=np.float32)
    elements_array = np.array([element for _, element in coords], dtype=h5py.string_dtype(encoding='utf-8'))

    with h5py.File(h5_filepath, 'w') as h5_file:
        h5_file.create_dataset("protein", data=protein_grid.astype(np.float32), dtype=np.float32)
        h5_file.create_dataset("pocket", data=pocket_label.astype(np.float32), dtype=np.float32)
        h5_file.create_dataset("coords", data=coords_array, dtype=np.float32)
        h5_file.create_dataset("elements", data=elements_array)

    logging.info(f"H5 file created for {protein_name}")


def generate_cache_for_proteins(protein_names, root_dir, cache_dir, max_workers=10):
    os.makedirs(cache_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_h5_file, protein_name, root_dir, cache_dir): protein_name
            for protein_name in protein_names
        }

        for future in as_completed(futures):
            protein_name = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing {protein_name}: {e}")


if __name__ == "__main__":
    config = load_config("./config/config.yml")
    data_dir = config.get("data_directory", "./data")
    cache_dir = config.get("cache_directory", "./cache")
    logging.info(f"Data directory: {data_dir}")
    logging.info(f"Cache directory: {cache_dir}")

    protein_names = config["datasets"]["train"] + config["datasets"]["validation"] + config["datasets"]["test"]

    generate_cache_for_proteins(protein_names, data_dir, cache_dir)
