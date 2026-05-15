# Work8 Design

Tarih: 2026-05-11

## Amac

Work8 tek bir ana calisma olarak tasarlandi. Daha once konustugumuz Work8A/Work8B ayrimi kaldirildi. Bu sayede raporda ve run klasorlerinde karmasa olmayacak.

Work8'in amaci:

APBS'in combined feature setlerdeki katkisini, birden fazla aday model ve birden fazla APBS representation ile olcmek.

## Tek Work8 Klasoru

Kullanilacak tek run klasoru:

`/Users/tevfik/Sandbox/github/PHD/runs/work8_combined_model_feature_representation_sweep_fold1_250epoch_thr040`

Eski yanlis baslayan klasor final sonuclara dahil edilmeyecek:

`/Users/tevfik/Sandbox/github/PHD/runs/work8_resnet3d4l_apbs_representation_combined_features_fold1_250epoch_thr040`

## Work8 Matrisi

Work8 bilincli olarak su matrisle sinirlandi:

```text
5 model x 2 feature set x 3 APBS representation = 30 training
```

Bu matris yeterince genis; cunku model-feature interaction'i gosterir. Ama tum olasi kombinasyonlari kosmadigi icin yorumlanabilir kalir.

## Modeller

Work8'de kullanilacak modeller:

- `ResNet3D4L`
- `UNet3D4LA`
- `UNetPlusPlus3D`
- `CBAMUNet3D`
- `ResNet3D4LGN`

Secim gerekcesi:

- `ResNet3D4L`: Work5/Work7 APBS-only lideri.
- `UNet3D4LA`: guclu klasik U-Net varyanti.
- `UNetPlusPlus3D`: Work5'te guclu alternatif.
- `CBAMUNet3D`: attention tabanli aday, DVO tarafinda anlamli olabilir.
- `ResNet3D4LGN`: Work6'da stabil ve guclu yeni ResNet varyanti.

## Feature Setler

Work8'de kullanilacak feature setler:

- `apbs_shape`
- `apbs_shape_selected_chem`

Secim gerekcesi:

- `apbs_only` zaten Work5 ve Work7'de detayli calisildi.
- `shape_only` ve `shape_selected_chem` Work1 icinde karsilastirma olarak var.
- Work8'in asil sorusu APBS'in combined feature setlerde nasil davrandigi oldugu icin feature setler APBS iceren kombinasyonlarla sinirlandi.

## APBS Representation

Work8'de kullanilacak APBS representation setleri:

- `apbs_clip20_minmax`
- `apbs_full_signed`
- `apbs_posneg_clip20`

Secim gerekcesi:

- `apbs_clip20_minmax`: Work7 genel lideri.
- `apbs_full_signed`: Work7'de top-3 DCC ve DCA tarafinda guclu sinyal verdi.
- `apbs_posneg_clip20`: Work7'de DVO ve top-3 DCC acisindan umut verici.

`clip5` ve `clip10` Work7'de belirgin zayif kaldigi icin Work8'e dahil edilmedi.

## Calistirma Scripti

Work8 icin hazirlanan script:

`/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/3dunet-apbs/3dunet_configurable/scripts/run_work8_combined_model_feature_sweep.sh`

Bu script dogru degiskenleri kullanir:

- `MODELS`
- `FEATURE_SETS`
- `CUTOFF_VARIANTS`

Bu onemli; cunku `run_work8_top_models_feature_sweep.sh` script'i `MODEL_CLASS` veya `APBS_VARIANTS` degiskenlerini okumaz.

## Calistirma Komutu

```bash
cd /Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/3dunet-apbs/3dunet_configurable

OUTPUT_ROOT=/Users/tevfik/Sandbox/github/PHD/runs/work8_combined_model_feature_representation_sweep_fold1_250epoch_thr040
mkdir -p "$OUTPUT_ROOT"

EPOCHS=250 \
EARLY_STOPPING_PATIENCE=0 \
VALIDATION_THRESHOLD=0.40 \
OUTPUT_ROOT="$OUTPUT_ROOT" \
SKIP_COMPLETED=1 \
CLEAN_INCOMPLETE=1 \
scripts/run_work8_combined_model_feature_sweep.sh > "$OUTPUT_ROOT/master.log" 2>&1 &
```

Log takip komutu:

```bash
tail -f /Users/tevfik/Sandbox/github/PHD/runs/work8_combined_model_feature_representation_sweep_fold1_250epoch_thr040/master.log
```

## Beklenen Cevap

Work8 sonunda su sorulara cevap aranacak:

1. Combined feature setlerde en iyi model hangisi?
2. APBS representation etkisi combined feature setlerde APBS-only ile ayni mi?
3. `apbs_full_signed` veya `apbs_posneg_clip20`, shape/chem eklendiginde `clip20_minmax`'i gecebilir mi?
4. Attention tabanli modeller combined feature setlerde ResNet'i gecebilir mi?
5. DVO tarafinda APBS representation ve model secimi nasil etkiliyor?

## Raporlama

Work8 bittiginde run klasorune `WORK8_RESULTS.md` eklenecek. Rapor su metrikleri icerecek:

- Selection score
- Pocket-F1
- DCC@4A
- Top-3 DCC@4A
- DCA@4A
- DVO(success)
- voxel-F1
- fixed-threshold F1@0.40

