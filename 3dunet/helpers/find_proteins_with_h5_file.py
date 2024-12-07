import os

# Dizini belirtiyoruz
directory = "/Volumes/Data/DiskYedek/DATA/dest/pdbbind-limited"

# .h5 dosyalarını içeren klasörleri depolamak için bir liste
folders_with_h5 = []

# Dizin içindeki tüm klasörleri ve dosyaları tarıyoruz
for folder in os.listdir(directory):
    folder_path = os.path.join(directory, folder)
    if os.path.isdir(folder_path):  # Sadece klasörleri kontrol et
        # Klasördeki dosyaları kontrol et
        for file in os.listdir(folder_path):
            if file.endswith(".h5"):  # Eğer .h5 dosyası varsa
                folders_with_h5.append(folder)
                break  # Bu klasörde bir tane bulduysak, diğerlerini kontrol etmeye gerek yok

# Sonuçları yazdır
print(f"Klasörlerde toplam {len(folders_with_h5)} h5 dosyasi bulundu...")
for folder in folders_with_h5:
    print(folder)