import os
import numpy as np

# Dizini belirtiyoruz
directory = "/Volumes/Data/DiskYedek/DATA/dest/pdbbind-limited"

# Dosyaları içeren klasörleri depolamak için bir liste
files_info = []

# Dizin içindeki tüm klasörleri ve dosyaları tarıyoruz
for folder in os.listdir(directory):
    folder_path = os.path.join(directory, folder)
    if os.path.isdir(folder_path):  # Sadece klasörleri kontrol et
        for file in os.listdir(folder_path):
            if file.endswith("_cache_grids.h5"):  # Belirtilen dosya adını kontrol et
                print("Deleting file {}".format(file))
                os.remove(os.path.join(folder_path, file))


# Sonuçları yazdır
print(f"Klasörlerde toplam {len(files_info)} '_grid.npy' dosyası bulundu.")
for info in files_info:
    print(
        f"Klasör: {info['folder']}, Dosya: {info['file_name']}, "
        f"Şekil: {info['shape']}, Min: {info['min_value']}, "
        f"Max: {info['max_value']}, Ortalama: {info['mean_value']:.2f}"
    )