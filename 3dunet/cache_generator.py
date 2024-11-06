import os

import h5py
import numpy as np
import yaml

from dataset import ProteinLigandDataset

# Yaml dosyasını yükleme
with open("config/config.yml", "r") as file:
    config = yaml.safe_load(file)

data_directory = config["data_directory"]


# H5 dosyasını oluşturma fonksiyonu
def create_h5_file(protein_name, protein_path, pocket_path):
    # Dosya adı protein adından oluşur
    h5_filename = f"{protein_name}.h5"
    h5_filepath = os.path.join(data_directory, "cache", h5_filename)

    # Eğer h5 dosyası zaten varsa, atla
    if os.path.exists(h5_filepath):
        print(f"{h5_filename} already exists. Skipping...")
        return

    # Protein ve pocket verilerini grid olarak elde et
    dataset = ProteinLigandDataset(root_dir=data_directory, protein_names=[protein_name])
    protein_grid = dataset.pdb_to_grid(protein_path)
    pocket_label = dataset.pdb_to_grid(pocket_path)

    # H5 dosyasına yaz
    with h5py.File(h5_filepath, "w") as h5_file:
        h5_file.create_dataset("protein", data=protein_grid, dtype=np.float32)
        h5_file.create_dataset("pocket", data=pocket_label, dtype=np.float32)
        print(f"Created {h5_filepath}")


# H5 dosyalarını oluştur
for dataset_type, proteins in config["datasets"].items():
    for protein_name in proteins:
        protein_path = os.path.join(data_directory, protein_name, f"{protein_name}_protein.pdb")
        pocket_path = os.path.join(data_directory, protein_name, f"{protein_name}_pocket.pdb")

        create_h5_file(protein_name, protein_path, pocket_path)
