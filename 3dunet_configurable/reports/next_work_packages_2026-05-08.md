# Next Work Packages - 2026-05-08

Bu not, son konuşmada toparlanan iş paketlerini ve çalıştırma komutlarını kaybetmemek için repo içine kaydedildi.

## Genel Durum

- APBS-only artık ana araştırma eksenlerinden biri. Sadece zayıf baseline gibi ele alınmamalı.
- Work3 sonucunda APBS `clip20` varyantı, `clip10` ve no-cutoff varyantlarından daha iyi göründü.
- Work4 sonucunda APBS `clip20`, kombine feature setlerinde de katkı sağladı.
- Work5 model sweep sonucunda tamamlanan modeller içinde `ResNet3D4L`, APBS-only `clip20` için en güçlü aday göründü.
- Ağır `ConvNeXtUNet3D` çok yavaş ilerlediği ve erken metrikleri zayıf olduğu için kullanıcı tarafından durduruldu.

## Guncel Kalan Work Sirasi - 2026-05-12

Bu sira 2026-05-12 tarihinde APBS/atomic feature cache iyilestirmesi konusmasindan sonra guncellendi. Bundan sonra ana plan olarak bu sira esas alinacak.

| Work | Durum | Amac |
|---|---|---|
| Work8 | Devam ediyor / analiz edilecek | Mevcut legacy cache uzerinde combined model-feature-APBS representation sweep |
| Work9 | Planlandi | BU48, COACH420 ve PDBBind external benchmark veri hazirligi/protokol netlestirme |
| Work10 | Opsiyonel/paralel | En iyi aday icin kucuk hyperparameter tuning |
| Work11 | Guncellendi | scPDB/PDBBind icin 36/72/161 cache; legacy feature'lari koruyup v2 APBS ve MOL2 atomic feature'lari ekleme |
| Work12 | Yeni eklendi | Work11 v2 feature'larinin ablation'i: legacy APBS vs v2 full-protein APBS, PDB atomic vs MOL2 atomic |
| Work13 | Work11/12 sonrasi | Full folded 36/72/161 Codon training |
| Work14 | Paralel gelistirilebilir | Protein-only prediction CLI ve 3D web analiz sayfasi |
| Work15 | Work13 sonrasi | BU48/COACH420/PDBBind external benchmark evaluation |
| Work16 | Final model sonrasi | Threshold ve postprocess calibration |
| Work17 | Son analiz | Error analysis, APBS interpretation, vaka bazli gorsel kanitlar |

Kritik karar: Work11 artik sadece cache boyutu uretimi degildir. Ayni zamanda eski feature'lari bozmadan yeni feature katmani ekleyen veri-muhendisligi isidir. Work12 bu yeni feature katmaninin model performansina etkisini olcecek ayri training isidir.

## Work5 Rapor Özeti

Work5 raporu şu dosyada:

```text
/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/3dunet-apbs/3dunet_configurable/reports/work5_model_sweep_report.md
```

Work5 tamamlanan modeller içinde ilk sonuçlar:

| Rank | Model | Selection | Pocket-F1 | DCC | DCA | DVO | Voxel-F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ResNet3D4L | 1.6599 | 0.6415 | 0.4722 | 0.7037 | 0.4888 | 0.5145 |
| 2 | UNet3D4LA | 1.5675 | 0.6154 | 0.4444 | 0.7130 | 0.4449 | 0.4785 |
| 3 | UNetPlusPlus3D | 1.5622 | 0.6154 | 0.4444 | 0.6667 | 0.4870 | 0.4730 |
| 4 | CBAMUNet3D | 1.5599 | 0.6154 | 0.4444 | 0.6852 | 0.4831 | 0.4696 |
| 5 | ResidualUNet3D | 1.5475 | 0.6154 | 0.4444 | 0.6852 | 0.4538 | 0.4623 |

Kısa yorum:

- `ResNet3D4L`, selection score, Pocket-F1, DCC, voxel-F1 ve fixed-threshold F1 açısından en iyi tamamlanan model.
- `SEResUNet3D`, DVO tarafında güçlüydü ama genel selection/Pocket-F1 tarafında `ResNet3D4L` kadar iyi değildi.
- Bu yüzden sonraki APBS representation deneylerinde default model olarak `ResNet3D4L` mantıklı.

## Work6 - Yeni Mimari Sweep

Durum: Tamamlandı.

Amaç: APBS-only `clip20` üzerinde, 36/72/161 gridlerde çalışabilecek daha makul mimarileri denemek.

Sonuç raporu:

```text
/Users/tevfik/Sandbox/github/PHD/runs/work6_apbs_only_clip20_new_architecture_sweep_fold1_250epoch_thr040/WORK6_RESULTS.md
```

Özet:

| Rank | Model | Selection | Pocket-F1 | DCC | DCA | DVO | Voxel-F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ResNet3D4LGN | 1.6340 | 0.6415 | 0.4722 | 0.7037 | 0.4687 | 0.4933 |
| 2 | CBAMResNet3D4LGN | 1.5868 | 0.6065 | 0.4352 | 0.7130 | 0.4808 | 0.4865 |
| 3 | SEResNet3D4LGN | 1.5754 | 0.6329 | 0.4630 | 0.7130 | 0.4419 | 0.4634 |
| 4 | TinyConvNeXtUNet3D | 1.4214 | 0.5974 | 0.4259 | 0.6296 | 0.4081 | 0.4237 |

Ana yorum:

- Yeni mimariler Work5'teki `ResNet3D4L` APBS-only liderini geçemedi.
- `ResNet3D4LGN`, Pocket-F1/DCC/DCA açısından Work5 liderine yaklaştı ama DVO ve voxel-F1 tarafında geride kaldı.
- Mimari değişikliği tek başına büyük sıçrama getirmedi; APBS representation ve combined feature setler daha öncelikli görünüyor.

## Work7 - APBS Representation Sweep

Durum: Tamamlandı.

Amaç: Work5'in en iyi modeli olan `ResNet3D4L` ile farklı APBS normalizasyonlarını karşılaştırmak.

Sonuç raporu:

```text
/Users/tevfik/Sandbox/github/PHD/runs/work7_apbs_representation_resnet3d4l_fold1_250epoch_thr040/WORK7_RESULTS.md
```

Özet:

| Rank | APBS representation | Selection | Pocket-F1 | DCC | DCA | DVO | Voxel-F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | apbs_clip20_minmax | 1.6599 | 0.6415 | 0.4722 | 0.7037 | 0.4888 | 0.5145 |
| 2 | apbs_full_signed | 1.6390 | 0.6329 | 0.4630 | 0.7315 | 0.4775 | 0.4959 |
| 3 | apbs_posneg_clip20 | 1.6359 | 0.6415 | 0.4722 | 0.6852 | 0.4923 | 0.4935 |
| 4 | apbs_no_cutoff_current | 1.6309 | 0.6329 | 0.4630 | 0.6944 | 0.4803 | 0.4980 |

Ana yorum:

- APBS-only için en güvenli temsil halen `apbs_clip20_minmax`.
- `apbs_full_signed` ve `apbs_posneg_clip20`, özellikle top-3 DCC/DVO tarafında sinyal taşıyor.
- `clip5` ve `clip10` zayıf kaldı; yüksek mutlak elektrostatik potansiyel bölgeleri bilgi taşıyor olabilir.

## Work8 - Combined Model/Feature/APBS Representation Sweep

Durum: Planlandı / çalıştırılacak tek Work8.

Amaç: APBS'in combined feature setlerdeki katkısını birden fazla aday model ve birden fazla APBS representation ile ölçmek.

Tasarım notu:

```text
/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/3dunet-apbs/3dunet_configurable/reports/work8_design_2026-05-11.md
```

Çalıştırma scripti:

```text
/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/3dunet-apbs/3dunet_configurable/scripts/run_work8_combined_model_feature_sweep.sh
```

Tek Work8 matrisi:

```text
5 model x 2 feature set x 3 APBS representation = 30 training
```

Modeller:

- `ResNet3D4L`
- `UNet3D4LA`
- `UNetPlusPlus3D`
- `CBAMUNet3D`
- `ResNet3D4LGN`

Feature setler:

- `apbs_shape`
- `apbs_shape_selected_chem`

APBS representation setleri:

- `apbs_clip20_minmax`
- `apbs_full_signed`
- `apbs_posneg_clip20`

Çalıştırma komutu:

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

Takip:

```bash
tail -f /Users/tevfik/Sandbox/github/PHD/runs/work8_combined_model_feature_representation_sweep_fold1_250epoch_thr040/master.log
```

Beklenen cevap:

- Combined feature setlerde en iyi model hangisi?
- `full_signed` veya `posneg_clip20`, shape/chem ile birleşince `clip20_minmax`'i geçiyor mu?
- Attention/UNet++/ResNet ailesi combined feature setlerde farklı davranıyor mu?
- APBS DVO ve top-3 DCC tarafında tutarlı katkı sağlıyor mu?

## Work9 - External Benchmark Data Preparation

Durum: Planlandı.

Amaç: BU48, COACH420 ve PDBBind external benchmark datasini final evaluation icin hazir hale getirmek.

Bu work package final evaluation'in kendisi degil, final evaluation protokol ve cache hazirligidir. Final external evaluation guncel sirada Work15'e tasindi.

Kapsam:

- BU48 raw data lokasyonlarini ve ligand/protein dosyalarini netlestirmek.
- COACH420 raw data lokasyonlarini ve ligand/protein dosyalarini netlestirmek.
- PDBBind external/held-out split adaylarini belirlemek.
- fpocket/P2Rank/ligand-derived reference pocket protokolunu yazmak.
- Work11 cache generator ile uyumlu benchmark cache uretim akisini hazirlamak.
- Test threshold'unun external testte optimize edilmeyecegini dokumante etmek.

Beklenen çıktı:

```text
/Users/tevfik/Sandbox/github/PHD/data/work9_external_benchmark_cache_*
```

## Work10 - Hyperparameter Optimization

Durum: Work8/Work9 sonrasında opsiyonel ama güçlü follow-up.

Amaç: En iyi model/feature/APBS representation kombinasyonu sabit tutulduktan sonra eğitim ve postprocess ayarlarını optimize etmek.

Önerilen küçük matris:

| Deneme | Learning rate | pos_weight | BCE/Dice alpha | Not |
|---:|---:|---:|---:|---|
| 1 | 1e-4 | 1 | 0.5 | baseline |
| 2 | 3e-4 | 1 | 0.5 | daha hızlı öğrenme |
| 3 | 5e-5 | 1 | 0.5 | daha stabil öğrenme |
| 4 | 1e-4 | 2 | 0.5 | hafif positive weighting |
| 5 | 1e-4 | 5 | 0.5 | daha güçlü positive weighting |
| 6 | 1e-4 | 1 | 0.3 | BCE ağırlığı artar |
| 7 | 1e-4 | 1 | 0.7 | Dice ağırlığı artar |
| 8 | 1e-4 | 2 | 0.7 | recall/overlap dengesi |

Ek postprocess hyperparameter adayları:

- `min_component_volume_angstrom3`: 25, 50, 100
- top-1 vs top-3 reporting
- validation-selected threshold vs fold-median threshold

Not:

- Work10, Work9'un yerini almaz.
- Work10'un amacı final modeli ince ayar yapmak; Work9'un amacı dış veri seti genellemesini göstermek.

## Work11 - scPDB/PDBBind 36/72/161 Cache Generation

Durum: Planlandı / cache repo içinde script ve notları hazırlandı / v2 feature kapsamı eklendi.

Amaç: scPDB ve PDBBind için aynı H5 şemasıyla 36, 72 ve 161 grid cache üretimini lokalde doğrulayıp Codon/SLURM üzerinde full üretmek. Default grid matrix `36:70 72:160 161:160` olacak. Eski feature isimleri korunacak, ayni H5 icine iyilestirilmis APBS ve MOL2 atomic v2 feature'lari eklenecek.

Detaylı Work11 dokümanı:

```text
/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/generate_cache/WORK11_CACHE_GENERATION.md
```

Local 36 script:

```text
/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/generate_cache/scripts/run_work11_local_box36_cache.sh
```

Codon/SLURM 36/72/161 script:

```text
/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/generate_cache/scripts/run_work11_slurm_cache_matrix.sbatch
```

Önemli düzeltme:

- 36 cache üretiminde görülen çok sayıda hata grid boyutundan değil, scPDB/PDBBind protein hazırlığında PDB2PQR/APBS'e hidrojenli protein verilmesinden kaynaklanıyordu.
- Gridfix cache kodu artık protein feature üretimi ve PDB2PQR girdisi için standart amino-acid heavy atom setini kullanıyor.
- Yeni H5 dosyalarında `protein_atom_policy = standard_amino_acid_heavy_atoms_only` attribute'u yazılıyor.
- V2 APBS icin default kaynak `ligand-free full clean protein` olacak; legacy/current APBS davranisi `selected_chains` adi ile ayri saklanacak.
- V2 atomic feature'lar MOL2 tabanli `atomic_mol2_*` kanallari olarak yazilacak.
- Manifest/H5 attrs icinde atom sayilari, dropped atom sebepleri, APBS source ve atomic source audit bilgileri yazilacak.

Work11 sırası:

1. Local 36 smoke cache: `LIMIT=20`.
2. Legacy+v2 feature varligi ve atom audit kontrolu.
3. Local 36 full cache: `LIMIT=all`.
4. Codon full matrix: `GRID_SPECS="36:70 72:160 161:160"`.
5. Üretim sonrası her klasörde `generation.log`, `manifest.csv` ve `failed_cases.txt` incelenecek.
6. Başarısız case'ler PDB2PQR, APBS, missing file ve label conversion hatası olarak kategorize edilecek.

## Work12 - V2 APBS and MOL2 Atomic Feature Ablation

Durum: Yeni eklendi.

Detaylı plan:

```text
/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/3dunet-apbs/3dunet_configurable/reports/work12_v2_feature_ablation_plan_2026-05-12.md
```

Amaç: Work11 ile ayni H5'e eklenen v2 feature'larin gercek model katkisini olcmek.

Ana sorular:

- Legacy `electrostatic_grid` mi, `electrostatic_grid_v2_full_protein_*` mi daha iyi?
- APBS icin `clip20_minmax`, `signed`, `posneg` temsilinden hangisi daha guvenli?
- PDB-derived legacy `atomic_*` kanallari mi, MOL2 tabanli `atomic_mol2_*` kanallari mi daha iyi?
- APBS + MOL2 chemistry, APBS-only veya APBS+shape uzerine anlamli katkı veriyor mu?
- `dist_to_surface_v2_full_protein` DVO'yu artiriyor mu?

Ilk matris:

```text
ResNet3D4L x 11 feature set x fold1 x 250 epoch
```

Ilk Work12 36 gridde kosulmali. En iyi 3-5 feature set Work13 full folded 36/72/161 training'e tasinmali.

## Work13 - Folded 36/72/161 Codon Training

Durum: Work11 full cache tamamlandıktan sonra yapılacak ana training işi.

Detaylı plan:

```text
/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/3dunet-apbs/3dunet_configurable/reports/work12_work13_plan_2026-05-11.md
```

Ana fikir:

- Limitli scPDB verisi sadece yön tayini ve bug yakalama için yeterli.
- Final skorlar için folded full scPDB/PDBBind çalışması gerekli.
- Default grid matrix: `36:70 72:160 161:160`.
- scPDB foldları base PDB id seviyesinde gruplanmalı.
- Her grid size için label coverage manifest üzerinden kontrol edilmeli.
- İlk güçlü matris: `apbs_only`, `shape_only`, `apbs_shape`, `shape_selected_chem`, `apbs_shape_selected_chem`.

Önemli not:

- `161:160` downscale değildir; APBS kaynak gridinin 1 A doğal temsilidir.
- Downsample olan asıl grid `72:160`; `36:70` ise daha küçük fiziksel alan ve 2 A spacing ile Kalasanty/PUResNet tarzına daha yakın hızlı gridtir.

## Work14 - Prediction Script and Web Analysis Page

Durum: Work13 ile paralel tasarlanabilir; final checkpoint Work13 sonrası seçilecek.

Amaç:

- Protein-only prediction için pocket tahmini.
- Ligand/known-pocket varsa DCC, DCA, DVO ve Pocket-F1 analizi.
- 3D görselleştirme, top-N pocket seçimi, threshold/postprocess kontrolleri.

Önerilen yapı:

- CLI: `predict_pocket.py`
- Backend: FastAPI
- Frontend: React/Vite + 3Dmol.js veya Mol*
- Output: `prediction.json`, pocket PDB files, metrics CSV, optional screenshot/export.

## Work15 - External Benchmark

Durum: Work13/Work14 altyapısı oturduktan sonra.

Amaç:

- BU48
- COACH420
- PDBBind external split

Bu work package literatür karşılaştırması için gerekli dış doğrulamayı sağlar.

## Work16 - Threshold/Postprocess Calibration

Durum: Work13/Work15 sonrası final model adayı seçilince.

Amaç:

- Final deploy threshold.
- Connected component postprocess.
- Top-1/top-3 reporting.
- Min component volume.

Kural:

- Threshold test datasında optimize edilmeyecek.
- Validation veya calibration split ile sabitlenecek.

## Work17 - Error Analysis and APBS Interpretation

Durum: Work13/Work15 sonuçlarından sonra.

Amaç:

- APBS'in hangi proteinlerde yardımcı olduğunu göstermek.
- DVO artışının vaka bazlı görsel kanıtını üretmek.
- 36/72/161 grid farklarını görsel olarak açıklamak.

## Local Cache Matrix - scPDB ve PDBBind

Amaç: 36, 72 ve 161 gridler için lokalde küçük veriyle cache pipeline'ını test etmek.

Script:

```text
/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/generate_cache/scripts/run_local_gridfix_cache_matrix.sh
```

Komut:

```bash
cd /Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/generate_cache

LIMIT=20 \
NPROC=4 \
BOX_SIZES="36 72 161" \
scripts/run_local_gridfix_cache_matrix.sh
```

Not:

- `LIMIT=20` sadece smoke/local test için.
- Full cache üretiminde limit kaldırılmalı.

## SLURM Full Cache Matrix

Amaç: scPDB ve PDBBind için full cache üretimini cluster üzerinde yapmak.

Script:

```text
/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/generate_cache/scripts/run_slurm_gridfix_cache_matrix.sbatch
```

Codon üzerinde örnek komut:

```bash
cd /homes/tevfik/PHD/phd_examples/generate_cache

sbatch scripts/run_slurm_gridfix_cache_matrix.sbatch
```

Notlar:

- 36, 72 ve 161 gridlerin tamamı üretilebilir.
- SLURM tarafında büyük dataset ve disk alanı daha uygun.
- Daha önce bazı proteinlerde cache üretimi başarısız olmuştu; bu scriptlerin logları sonrasında ayrıca analiz edilmeli.

## BU48 ve COACH Hazırlığı

Amaç: PUResNet/Kalasanty karşılaştırması için BU48 ve COACH420 datasetlerini hazır hale getirmek.

Hazırlama scripti:

```text
/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/generate_cache/scripts/run_prepare_puresnet_benchmarks.sh
```

Cache üretim scripti:

```text
/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/generate_cache/scripts/run_local_puresnet_benchmark_cache_matrix.sh
```

Örnek lokal komut:

```bash
cd /Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/generate_cache

scripts/run_prepare_puresnet_benchmarks.sh

LIMIT=20 \
NPROC=4 \
BOX_SIZES="36 72 161" \
scripts/run_local_puresnet_benchmark_cache_matrix.sh
```

Full BU48/COACH cache için:

```bash
cd /Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/generate_cache

scripts/run_prepare_puresnet_benchmarks.sh

NPROC=4 \
BOX_SIZES="36 72 161" \
scripts/run_local_puresnet_benchmark_cache_matrix.sh
```

## Eklenmiş Kod Değişiklikleri

Bu konuşma sonunda repo içinde şu pratik değişiklikler vardı:

- `main.py`: validation summary içine model adı eklendi.
- `dataset.py`: APBS için signed normalization ve pozitif/negatif kanal desteği eklendi.
- `models/ModernUNet3D.py`: yeni 3D modeller eklendi:
  - `ResNet3D4LGN`
  - `SEResNet3D4LGN`
  - `CBAMResNet3D4LGN`
  - `TinyConvNeXtUNet3D`
- `scripts/run_apbs_cutoff_sweep.sh`: APBS varyantları genişletildi.
- Work6/Work7/Work8 scriptleri eklendi.
- Work8 tek çalışma olacak şekilde yeniden tasarlandı:
  - `scripts/run_work8_combined_model_feature_sweep.sh`
  - `reports/work8_design_2026-05-11.md`
- BU48/COACH ve cache matrix scriptleri eklendi.

## Doğrulama

Yapılan hızlı doğrulamalar:

- `py_compile` geçti:
  - `main.py`
  - `dataset.py`
  - `models/ModernUNet3D.py`
  - `scripts/write_work5_model_sweep_report.py`
  - `generate_cache_benchmark_gridfix.py`
- `bash -n` yeni shell scriptlerinde geçti.
- Yeni modeller için dummy forward testleri 36, 72 ve 161 input size üzerinde geçti.
- Work6/Work7 dry-run config üretimleri geçti ve gerçek koşular tamamlandı.
- Work8 yeni wrapper scripti için `bash -n` geçti.

## Kritik Notlar

- `generate_cache_benchmark_gridfix.py` gerçek BU48/COACH datası üzerinde henüz bu konuşmada çalıştırılmadı; syntax ve script yapısı kontrol edildi.
- Work6 ve Work7 gerçek training koşuları tamamlandı ve raporlandı.
- Work8 yeni tek plan olarak hazırlanmıştır; gerçek training kullanıcı tarafından başlatılacak.
- Bundan sonra uzun training veya cache üretimi başlatmadan önce kullanıcıdan açık onay alınmalı.
- Eski/yarım kalan loglar sonuç analizinde karışıklık yaratabilir; `run_summary.csv` ve run bazlı klasörler öncelikli kaynak olmalı.

## Eksik veya Atlanmaması Gereken Noktalar

1. Work8 sonuçları bitince:
   - APBS-only ile APBS+shape ve APBS+selected_chem aynı model ailesinde karşılaştırılmalı.
   - APBS'in DVO katkısı özel olarak raporlanmalı.
   - `clip20_minmax`, `full_signed` ve `posneg_clip20` combined feature setlerde karşılaştırılmalı.
   - Top-1 DCC ve top-3 DCC birlikte raporlanmalı.

2. Work9 external benchmark hazırlığında:
   - BU48/COACH label/pocket seçim protokolü netleştirilmeli.
   - Prediction threshold testte optimize edilmemeli; validation-selected veya fold-median threshold kullanılmalı.
   - Literatürle karşılaştırmada DCC/DCA/DVO/Pocket-F1 protokolü açık yazılmalı.

3. Work10 hyperparameter tuning başlatılırsa:
   - Work8/Work9'dan önce değil, en iyi model/feature kombinasyonu seçildikten sonra yapılmalı.
   - Aynı anda çok fazla parametre değiştirilmemeli.
   - Her koşuda tek ana değişken değiştirilmeli.

4. Full cache tamamlanınca:
   - Başarısız proteinler listelenmeli.
   - Başarısızlık sebepleri kategorize edilmeli.
   - scPDB için Kalasanty benzeri filtreleme ve full-dataset yaklaşımı ayrı tutulmalı.

5. BU48/COACH için:
   - P2Rank/PUResNet benchmark yaklaşımıyla label/pocket seçim mantığı netleştirilmeli.
   - fpocket selected pocket kullanımı ve ligand merkezine göre seçim açıkça dokümante edilmeli.

6. Makale/tez açısından:
   - Ana iddia şimdilik "APBS-based electrostatic potential representation improves or complements 3D binding-site prediction under controlled feature/model ablations" çizgisinde tutulmalı.
   - "İlk kez" gibi güçlü novelty iddiası dikkatli kurulmalı; Deep-APBS önceki grup çalışması olarak doğru konumlandırılmalı.
   - APBS-only güçlü çıkarsa bu zaten tek başına önemli bir bulgu olabilir.
