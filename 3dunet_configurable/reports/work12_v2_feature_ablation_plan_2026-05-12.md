# Work12 - V2 APBS and MOL2 Atomic Feature Ablation

Bu work package Work11 cache guncellemesinden sonra calistirilacak. Amac, eski feature'lari koruyarak uretilen yeni APBS ve MOL2 atomic feature'larin modele gercek katkisini kontrollu olarak olcmektir.

## Neden Ayrı Work?

Work11 veri uretim ve audit isidir. Work12 ise model davranisini olcer. Bu ikisini ayirmak gerekir:

- Cache dogru mu?
- Atom setleri bekledigimiz gibi mi?
- APBS full-protein grid eski selected-chain APBS'ten iyi mi?
- MOL2 atomic feature'lar PDB-derived atomic feature'lardan iyi mi?
- APBS + shape + MOL2 chemistry birlikte DCC/DVO/Pocket-F1'i yukari tasiyor mu?

Bu sorular training/evaluation sorularidir; cache uretiminden ayri raporlanmalidir.

## Gerekli Cache

Work12 baslamadan once Work11'den en az bir dogrulanmis cache seti hazir olmalidir:

```text
box36_span70
```

Ilk ablation icin 36 grid yeterlidir. 72 ve 161 gridler Work13 folded full training icin daha uygundur.

## Legacy Feature Setleri

Eski feature'lar aynen korunur ve baseline olarak kosulur:

```text
legacy_apbs_only:
  features/electrostatic_grid

legacy_shape_only:
  features/shape

legacy_shape_selected_chem:
  features/shape
  features/atomic_C
  features/atomic_N
  features/atomic_O
  features/atomic_S
  features/atomic_hydrophobic
  features/atomic_aromatic
  features/atomic_acceptor
  features/atomic_donor

legacy_apbs_shape:
  features/electrostatic_grid
  features/shape

legacy_apbs_shape_selected_chem:
  features/electrostatic_grid
  features/shape
  selected legacy atomic channels
```

## V2 Feature Setleri

Yeni feature setleri:

```text
v2_apbs_full_raw_only:
  features/electrostatic_grid_v2_full_protein_raw

v2_apbs_full_clip20_only:
  features/electrostatic_grid_v2_full_protein_clip20_minmax

v2_apbs_full_signed_only:
  features/electrostatic_grid_v2_full_protein_signed

v2_apbs_full_posneg_only:
  features/electrostatic_grid_v2_full_protein_pos
  features/electrostatic_grid_v2_full_protein_neg

v2_apbs_selected_chains_only:
  features/electrostatic_grid_v2_selected_chains_raw

v2_shape_only:
  features/shape_v2_full_protein

v2_mol2_selected_chem:
  features/atomic_mol2_C
  features/atomic_mol2_N
  features/atomic_mol2_O
  features/atomic_mol2_S
  features/atomic_mol2_hydrophobic
  features/atomic_mol2_aromatic
  features/atomic_mol2_acceptor
  features/atomic_mol2_donor

v2_apbs_shape:
  features/electrostatic_grid_v2_full_protein_clip20_minmax
  features/shape_v2_full_protein

v2_apbs_mol2_selected_chem:
  features/electrostatic_grid_v2_full_protein_clip20_minmax
  selected atomic_mol2 channels

v2_apbs_shape_mol2_selected_chem:
  features/electrostatic_grid_v2_full_protein_clip20_minmax
  features/shape_v2_full_protein
  selected atomic_mol2 channels

v2_apbs_shape_mol2_selected_chem_surface:
  features/electrostatic_grid_v2_full_protein_clip20_minmax
  features/shape_v2_full_protein
  selected atomic_mol2 channels
  features/dist_to_surface_v2_full_protein
```

## Model Secimi

Ilk Work12 kosusu icin model sabit tutulmali:

```text
ResNet3D4L
```

Neden: Work5 ve Work7'de APBS-only icin en guclu tamamlanan model buydu. Bu ablation'da mimari degil feature etkisi olculmeli.

Ikinci fazda gerekirse:

```text
ResNet3D4LGN
UNet3D4LA
UNetPlusPlus3D
```

## Ilk Kosu Matrisi

Pratik ilk matris:

```text
1 model x 11 feature set x 1 fold x 200-250 epoch
```

Ilk fold:

```text
fold1
```

Epoch:

```text
250
```

APBS gec ogrenebildigi icin 150 epoch erken kalabilir.

## Raporlanacak Metrikler

Ana tablo:

```text
selection_score
Pocket-F1
DCC@4A
DCA@4A
DVO_success
DVO_all
voxel-F1
fixed voxel-F1@0.40
best validation threshold
best epoch
mean predicted pocket size
no-prediction count
top-1 DCC@4A
top-3 DCC@4A
```

Literature comparison icin ana metrikler:

```text
Top-1 Pocket-F1
Top-1 DCC@4A
DCA@4A
DVO_success
```

Top-3 sonucu ayrica diagnostic olarak verilecek; PUResNet/Kalasanty ile direkt ayni metrik gibi sunulmayacak.

## Beklenen Cevaplar

Work12 sonunda su kararlar verilmeli:

1. Legacy APBS mi, full-protein APBS v2 mi daha iyi?
2. APBS icin `clip20_minmax`, `signed`, `posneg` arasinda hangisi daha guvenli?
3. MOL2 atomic feature'lar PDB-derived legacy atomic feature'lardan iyi mi?
4. APBS + MOL2 chemistry, APBS-only'e anlamli katkı veriyor mu?
5. `dist_to_surface_v2_full_protein` DVO'yu artiriyor mu yoksa modeli shape-heavy hale mi getiriyor?
6. Work13 full folded 36/72/161 training'e hangi 3-5 feature set tasinmali?

## Basari Kriteri

Bu work icin basari sadece en yuksek tek skor degil. Basarili sayilmasi icin:

- v2 feature setlerden en az biri legacy karsiligini DCC/Pocket-F1 veya DVO_success tarafinda gecmeli.
- APBS-only v2, legacy APBS-only'den daha anlamli veya daha stabil olmali.
- Atom audit loglari feature farkini aciklayabilecek kadar temiz olmali.

