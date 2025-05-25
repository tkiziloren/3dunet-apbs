import yaml
import random

# --- KULLANICI AYARLARI ---
input_yaml = "config/codon/pdbbind_optimized_pos_weight_1_batch_size_4.yml"
output_yaml = "config/codon/pdbbind_randomized_config.yml"
test_ratio = 0.1
val_ratio = 0.1
random_seed = 42
# --------------------------

# 1. Var olan config dosyasını yükle
with open(input_yaml, "r") as f:
    config = yaml.safe_load(f)

# 2. Tüm protein isimlerini tek bir listede topla
protein_sets = []
if "datasets" in config:
    for section in ["train", "validation", "test"]:
        proteins = config["datasets"].get(section)
        if proteins is not None:
            protein_sets.extend(proteins)
else:
    raise ValueError("Config dosyasında 'datasets' anahtarı yok.")

all_proteins = list(set(protein_sets))
random.Random(random_seed).shuffle(all_proteins)

# 3. Split oranlarını uygula
total = len(all_proteins)
test_count = int(total * test_ratio)
val_count = int(total * val_ratio)
train_count = total - test_count - val_count

test = all_proteins[:test_count]
val = all_proteins[test_count:test_count+val_count]
train = all_proteins[test_count+val_count:]

# 4. Yeni config yapısı oluştur
new_config = config.copy()
new_config["datasets"]["train"] = sorted(train)
new_config["datasets"]["validation"] = sorted(val)
new_config["datasets"]["test"] = sorted(test)

# 5. Dosyaya yaz
with open(output_yaml, "w") as f:
    yaml.dump(new_config, f, default_flow_style=False)

print(f"Yeni random split config dosyası yazıldı: {output_yaml}")
print(f"Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")