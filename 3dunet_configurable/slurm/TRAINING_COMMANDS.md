# Feature Selection Experiments - Training Commands

## Kullanım

```bash
cd ~/PHD/3dunet-apbs/3dunet_configurable/slurm
bash train_new.sh JOB_NAME CONFIG_FILE MODEL_CLASS [GPU_COUNT] [GPU_TYPE] [CPUS] [BASE_FEATURES] [MEMORY_GB]
```

## Baseline Denemeleri

### Baseline - Sadece Electrostatic Grid
```bash
# Dataset label ile
bash train_new.sh baseline_dataset config/codon_with_different_selections/baseline_label_dataset_binding_site.yml UNet3D4L

# Calculated label ile  
bash train_new.sh baseline_calculated config/codon_with_different_selections/baseline_label_calculated_binding_site.yml UNet3D4L
```

## Single Feature Addition - Her feature'ı teker teker test et

### Shape
```bash
bash train_new.sh electrostatic_shape_dataset config/codon_with_different_selections/baseline_with_shape_label_calculated_binding_site.yml UNet3D4L
bash train_new.sh electrostatic_shape_calculated config/codon_with_different_selections/baseline_with_shape_label_dataset_binding_site.yml UNet3D4L
```

### Atomic Features
```bash
# Nitrogen
bash train_new.sh electrostatic_nitrogen_dataset config/codon_with_different_selections/electrostatic_nitrogen_label_dataset_binding_site.yml UNet3D4L
bash train_new.sh electrostatic_nitrogen_calculated config/codon_with_different_selections/electrostatic_nitrogen_label_calculated_binding_site.yml UNet3D4L

# Oxygen
bash train_new.sh electrostatic_oxygen_dataset config/codon_with_different_selections/electrostatic_oxygen_label_dataset_binding_site.yml UNet3D4L
bash train_new.sh electrostatic_oxygen_calculated config/codon_with_different_selections/electrostatic_oxygen_label_calculated_binding_site.yml UNet3D4L

# Carbon
bash train_new.sh electrostatic_carbon_dataset config/codon_with_different_selections/electrostatic_carbon_label_dataset_binding_site.yml UNet3D4L
bash train_new.sh electrostatic_carbon_calculated config/codon_with_different_selections/electrostatic_carbon_label_calculated_binding_site.yml UNet3D4L
```

### Chemical Features
```bash
# Donor
bash train_new.sh electrostatic_donor_dataset config/codon_with_different_selections/electrostatic_donor_label_dataset_binding_site.yml UNet3D4L
bash train_new.sh electrostatic_donor_calculated config/codon_with_different_selections/electrostatic_donor_label_calculated_binding_site.yml UNet3D4L

# Acceptor
bash train_new.sh electrostatic_acceptor_dataset config/codon_with_different_selections/electrostatic_acceptor_label_dataset_binding_site.yml UNet3D4L
bash train_new.sh electrostatic_acceptor_calculated config/codon_with_different_selections/electrostatic_acceptor_label_calculated_binding_site.yml UNet3D4L

# Hydrophobic
bash train_new.sh electrostatic_hydrophobic_dataset config/codon_with_different_selections/electrostatic_hydrophobic_label_dataset_binding_site.yml UNet3D4L
bash train_new.sh electrostatic_hydrophobic_calculated config/codon_with_different_selections/electrostatic_hydrophobic_label_calculated_binding_site.yml UNet3D4L

# Aromatic
bash train_new.sh electrostatic_aromatic_dataset config/codon_with_different_selections/electrostatic_aromatic_label_dataset_binding_site.yml UNet3D4L
bash train_new.sh electrostatic_aromatic_calculated config/codon_with_different_selections/electrostatic_aromatic_label_calculated_binding_site.yml UNet3D4L
```

## Meaningful Pairs

### Donor + Acceptor (Hydrogen bonding)
```bash
bash train_new.sh donor_acceptor_dataset config/codon_with_different_selections/electrostatic_donor_acceptor_label_dataset_binding_site.yml UNet3D4L
bash train_new.sh donor_acceptor_calculated config/codon_with_different_selections/electrostatic_donor_acceptor_label_calculated_binding_site.yml UNet3D4L
```

### Shape + Donor
```bash
bash train_new.sh shape_donor_dataset config/codon_with_different_selections/electrostatic_shape_donor_label_dataset_binding_site.yml UNet3D4L
bash train_new.sh shape_donor_calculated config/codon_with_different_selections/electrostatic_shape_donor_label_calculated_binding_site.yml UNet3D4L
```

### Hydrophobic + Aromatic
```bash
bash train_new.sh hydrophobic_aromatic_dataset config/codon_with_different_selections/electrostatic_hydrophobic_aromatic_label_dataset_binding_site.yml UNet3D4L
bash train_new.sh hydrophobic_aromatic_calculated config/codon_with_different_selections/electrostatic_hydrophobic_aromatic_label_calculated_binding_site.yml UNet3D4L
```

## Comprehensive Combinations

### All Chemical Features
```bash
bash train_new.sh all_chemical_dataset config/codon_with_different_selections/electrostatic_all_chemical_label_dataset_binding_site.yml UNet3D4L
bash train_new.sh all_chemical_calculated config/codon_with_different_selections/electrostatic_all_chemical_label_calculated_binding_site.yml UNet3D4L
```

### Full Context (Tüm feature'lar)
```bash
bash train_new.sh full_context_dataset config/codon_with_different_selections/electrostatic_full_context_label_dataset_binding_site.yml UNet3D4L
bash train_new.sh full_context_calculated config/codon_with_different_selections/electrostatic_full_context_label_calculated_binding_site.yml UNet3D4L
```

## Farklı Model Mimarileri ile Test

```bash
# UNet3D5L
bash train_new.sh electrostatic_shape_dataset_5L config/codon_with_different_selections/electrostatic_shape_label_dataset_binding_site.yml UNet3D5L

# ConvNeXt3D
bash train_new.sh electrostatic_shape_dataset_convnext config/codon_with_different_selections/electrostatic_shape_label_dataset_binding_site.yml ConvNeXt3D

# ResNet3D4L
bash train_new.sh electrostatic_shape_dataset_resnet config/codon_with_different_selections/electrostatic_shape_label_dataset_binding_site.yml ResNet3D4L
```

## Toplu İş Gönderimi

Tüm single feature testlerini göndermek için:

```bash
cd ~/PHD/3dunet-apbs/3dunet_configurable/slurm

for feature in baseline shape nitrogen oxygen carbon donor acceptor hydrophobic aromatic; do
  bash train_new.sh electrostatic_${feature}_dataset \
    config/codon_with_different_selections/electrostatic_${feature}_label_dataset_binding_site.yml \
    UNet3D4L
done
```

## Notlar

- **Feature sayısı** otomatik olarak config'den okunur, manuel belirtmeye gerek yok
- **BASE_FEATURES=64** varsayılan, model kapasitesi için
- **GPU_TYPE=h200** varsayılan, h200 yoksa a100 kullanın
- **MEMORY=128GB** varsayılan, büyük modeller için yeterli
- Her job için otomatik log dosyaları: `/hps/nobackup/arl/chembl/tevfik/3dunet-apbs/logs/`
- Model weights: `/hps/nobackup/arl/chembl/tevfik/3dunet-apbs/output/`
