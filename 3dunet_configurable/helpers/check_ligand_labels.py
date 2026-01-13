import os
import h5py
import numpy as np

# Ana dizin yolu
base_dir = "/Volumes/Data/DiskYedek/DATA/dest/pdbbind-limited"

# Tüm protein klasörlerini dolaş
for protein_dir in os.listdir(base_dir):
    protein_path = os.path.join(base_dir, protein_dir)
    grids_file = os.path.join(protein_path, f"{protein_dir}_grids.h5")

    # grids.h5 dosyasını kontrol et
    if os.path.isfile(grids_file):
        try:
            # H5 dosyasını oku
            with h5py.File(grids_file, "r") as h5_file:
                # Label matrisini yükle
                if "label" in h5_file:
                    label_matrix = h5_file["label"][:]

                    # Benzersiz değerleri kontrol et
                    unique_values = np.unique(label_matrix)
                    num_ones = np.sum(label_matrix == 1)
                    num_zeros = np.sum(label_matrix == 0)

                    # Oranları hesapla (0'a bölme kontrolü)
                    ratio = num_ones / num_zeros if num_zeros > 0 else np.inf

                    print(f"Protein: {protein_dir}, 1s: {num_ones}, 0s: {num_zeros}, 1/0 ratio: {ratio:.4f}")

                    print(f"Protein: {protein_dir}, Unique values in label matrix: {unique_values}")
               # else:
                    # print(f"Protein: {protein_dir}, 'label' dataset not found in {grids_file}")
        except Exception as e:
            print(f"Error reading {grids_file}: {e}")
    # else:
    #     print(f"Protein: {protein_dir}, grids.h5 file not found")