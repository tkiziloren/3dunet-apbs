# Consolidated Work Packages - 2026-05-18

Bu rapor, 2026-05-18 itibariyle APBS/electrostatics-aware 3D binding-site segmentation calismalarini tek siraya toplar. Eski raporlardaki Work numaralari korunmaya calisildi; yeni kararlar bu dosyada guncel plan olarak yazildi.

## Kapsam ve Kurallar

- Su anda devam eden Codon egitimleri bitmeden yeni uzun egitim baslatilmayacak.
- PDBbind ve scPDB sonuclari karistirilmayacak.
- Kalasanty foldlari scPDB foldlaridir; PDBbind icin kullanilmayacak.
- PUResNet-v1 scPDB benchmark final subseti 5020 komplekstir; public repo exact fold assignment vermedigi icin bizim deterministic 4-fold split "exact paper folds" diye raporlanmayacak.
- Paper-style DCC default'u predicted pocket center ile actual binding-site label center arasi mesafedir.
- DCA ligand mask/ligand atomlarina en yakin mesafe olarak kalir.
- DVO predicted pocket mask ile actual binding-site label mask IoU'sudur.
- PLI predicted pocket mask icindeki ligand mask kapsama oranidir.
- DVO ve PLI hem all-protein hem DCC-success subset uzerinden raporlanacak.
- Running job'lar eski metrik koduyla baslamis olabilir. Bitince yeni DCC/PLI koduyla post-hoc re-evaluation yapilacak.

## Guncel Kisa Oncelik

1. Devam eden PDBbind Work10/12 job'lari bitsin.
2. Work10/12/13 checkpoint'leri yeni DCC/PLI metrikleriyle yeniden degerlendirilsin.
3. scPDB box36 Kalasanty 10-fold ResNet3D4L benchmark calissin.
4. scPDB box36 PUResNet 5020 available-H5 deterministic 4-fold ResNet3D4L benchmark calissin.
5. Bu baseline'lar bittikten sonra ayni scPDB benchmarklarda `UNetPlusPlus3D` follow-up calissin.
6. Sonra sadece en iyi adaylar icin loss/Tversky ve modern model pilotlari yapilsin.

## Work 1 - Baseline Feature Ablation

Durum: Tamamlandi / legacy.

Amac: scPDB box36 uzerinde APBS, shape, selected chemistry ve bunlarin kombinasyonlarini ilk kez karsilastirmak.

Ana cikti:

- `shape_only` ve `shape_selected_chem` guclu baseline'lar olarak goruldu.
- APBS'nin tek basina zayif kalabildigi, ancak shape/chem ile birlikte anlamli olabilecegi anlasildi.
- Work2/Work4/Work8 feature aileleri buradan secildi.

## Work 2 - Box36 Top Feature Families + APBS Control

Durum: Tamamlandi / legacy.

Amac: Work1'den gelen en guclu feature ailelerini yeni split ile tekrar kosmak ve APBS-only kontrolunu korumak.

Kullanilan script:

```text
3dunet_configurable/scripts/run_work2_box36_top5_plus_apbs_newsplit.sh
```

Ana feature aileleri:

- `apbs_shape_selected_chem`
- `apbs_shape`
- `apbs_shape_selected_chem_surface_hydro`
- `shape_selected_chem`
- `shape_only`
- `apbs_only`

## Work 3 - APBS Cutoff Sweep

Durum: Tamamlandi.

Amac: APBS-only tarafta cutoff/normalization etkisini olcmek.

Ana sonuc:

- `clip20` varyanti, `clip10` ve no-cutoff varyantlarindan daha guvenli gorundu.
- APBS-only zayif bir baseline degil; fiziksel sinyal tasiyor ama tek basina final model icin yeterli degil.

## Work 4 - APBS clip20 Combined Feature Test

Durum: Tamamlandi.

Amac: Work3'te iyi gorunen `clip20` APBS temsilini combined feature setlerde denemek.

Kullanilan script:

```text
3dunet_configurable/scripts/run_work4_apbs_clip20_combined_features.sh
```

Ana sonuc:

- APBS + shape ve APBS + shape + selected chemistry kombinasyonlari, APBS-only'den daha anlamli hale geldi.
- Combined feature setlerde APBS katkisini daha buyuk Work8 matrisiyle test etmek mantikli hale geldi.

## Work 5 - APBS-only Model Sweep

Durum: Tamamlandi.

Kaynak rapor:

```text
3dunet_configurable/reports/work5_model_sweep_report.md
```

Amac: APBS-only `clip20` uzerinde model mimarisi etkisini olcmek.

Ana sonuc:

| Rank | Model | Selection | Pocket-F1 | DCC | DCA | DVO(success) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `ResNet3D4L` | 1.6599 | 0.6415 | 0.4722 | 0.7037 | 0.4888 |
| 2 | `UNet3D4LA` | 1.5675 | 0.6154 | 0.4444 | 0.7130 | 0.4449 |
| 3 | `UNetPlusPlus3D` | 1.5622 | 0.6154 | 0.4444 | 0.6667 | 0.4870 |

Yorum:

- APBS-only kosulda en guclu model `ResNet3D4L` idi.
- Bu, sonraki APBS representation ve PDBbind folded isleri icin ResNet3D4L'i ucuz ve stabil default yapti.

## Work 6 - New Architecture Sweep

Durum: Tamamlandi.

Amac: APBS-only `clip20` uzerinde yeni mimarileri denemek.

Ana sonuc:

- `ResNet3D4LGN` guclu ve stabil gorundu ama Work5 lideri `ResNet3D4L`'i net gecmedi.
- GroupNorm tek basina anlamli sicrama getirmedi.
- Mimari degisikliginden cok feature representation ve combined feature setler daha onemli gorundu.

## Work 7 - APBS Representation Sweep with ResNet3D4L

Durum: Tamamlandi.

Amac: `ResNet3D4L` ile APBS representation farkini olcmek.

Ana sonuc:

| Rank | APBS representation | Selection | Pocket-F1 | DCC | DCA | DVO(success) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `apbs_clip20_minmax` | 1.6599 | 0.6415 | 0.4722 | 0.7037 | 0.4888 |
| 2 | `apbs_full_signed` | 1.6390 | 0.6329 | 0.4630 | 0.7315 | 0.4775 |
| 3 | `apbs_posneg_clip20` | 1.6359 | 0.6415 | 0.4722 | 0.6852 | 0.4923 |

Yorum:

- APBS-only icin `clip20_minmax` en guvenliydi.
- Ancak `full_signed` ve `posneg_clip20` combined feature setlerde tekrar denenmeye deger sinyal tasidi.

## Work 8 - Combined Model/Feature/APBS Sweep

Durum: Tamamlandi.

Kaynak rapor:

```text
3dunet_configurable/reports/work8_results_2026-05-15.md
```

Matris:

```text
5 model x 2 feature set x 3 APBS representation = 30 training
```

Ana sonuc:

- En guclu model ailesi `UNetPlusPlus3D` oldu.
- En iyi genel kosu: `UNetPlusPlus3D + apbs_shape + apbs_full_signed`.
- En iyi DVO(success) kosusu: `UNetPlusPlus3D + apbs_shape_selected_chem + apbs_full_signed`.

En iyi kosu:

| Model | Feature | APBS | Selection | Pocket-F1 | DCC | DCA | DVO(success) |
|---|---|---|---:|---:|---:|---:|---:|
| `UNetPlusPlus3D` | `apbs_shape` | `full_signed` | 1.8376 | 0.7143 | 0.5556 | 0.7315 | 0.5354 |

Model ortalamalari:

| Model | Avg selection | Avg Pocket-F1 | Avg DCC | Avg DCA | Avg DVO(success) |
|---|---:|---:|---:|---:|---:|
| `UNetPlusPlus3D` | 1.8073 | 0.6894 | 0.5262 | 0.7485 | 0.5371 |
| `CBAMUNet3D` | 1.7485 | 0.6827 | 0.5185 | 0.7145 | 0.5000 |
| `UNet3D4LA` | 1.7444 | 0.6731 | 0.5077 | 0.7299 | 0.5096 |
| `ResNet3D4L` | 1.7301 | 0.6730 | 0.5077 | 0.7145 | 0.5162 |

Yorum:

- Combined feature setlerde en iyi U-Net adayi `UNetPlusPlus3D`.
- PDBbind/scPDB baseline islerinde `ResNet3D4L` maliyet/stabilite icin kullaniliyor, ama finalist mimari kontrolu icin `UNetPlusPlus3D` mutlaka tekrar denenmeli.

## Work 9 - Work8 Top-k Re-evaluation

Durum: Tamamlandi, fakat yeni metrik koduyla tekrar edilebilir.

Kaynaklar:

```text
3dunet_configurable/reports/work8_top1_top3_comparison.md
3dunet_configurable/reports/work8a_topk_metrics_2026-05-15/
```

Amac: Top-1, Top-3 ve Top-(n+2) cep degerlendirmesiyle literature-style metrikleri netlestirmek.

Onemli not:

- Bu is eski DCC reference ile uretilmis olabilir.
- Yeni `dcc_reference=label_center` ve PLI ile yeniden uretmek daha dogru olur.

## Work 10 - PDBbind box36 Hyperparameter Sweep

Durum: Codon'da devam ediyor.

Root:

```text
/nfs/production/arl/chembl/tevfik/DEEP_APBS_DATASETS/runs/work10_pdbbind_box36_span70_v1_hparam_selectedchem_24grid_150epoch_thr040_lightval30
```

Amac: `apbs_v1_full_signed_shape_selected_chem` uzerinde learning rate, loss alpha, pos_weight ve weight decay etkisini olcmek.

Su ana kadar guclu gorunen ayarlar:

- `lr1e4_alpha05_pos2_wd1e5`
- `lr2e4_alpha05_pos1_wd1e5`
- `lr1e4_alpha08_pos1_wd1e5`

Yorum:

- Yeni scPDB baseline icin kontrollu default olarak `lr1e-4 alpha0.5 pos_weight2 wd1e-5` kullanilabilir.
- Mevcut PDBbind sonucunun yeni DCC/PLI ile tekrar degerlendirilmesi gerekiyor.

## Work 11 - Cache Generation and Cache Audit

Durum: Buyuk olcude tamamlandi / aktif audit devam ediyor.

Amac: scPDB/PDBbind icin box36/box72/box161 H5 cache uretmek, feature/label/ligand mask konumlarini dogrulamak.

Onemli durum:

- scPDB work11 path: `cache/work11_cache_gridfix_v1/scpdb/label_cavity6/box36_span70`.
- H5'lerde ligand mask `features/ligand` altinda degil, `auxiliary/ligand` altinda.
- scPDB label source mevcut benchmark icin `binding_site_cavity6`.
- PUResNet 5020 check sonucunda H5 bulunmayan 207 case var; label atoms outside box bayragi ayri raporlanmali, otomatik dislama olarak kullanilmamali.

## Work 12 - PDBbind box36 Ablation

Durum: Codon'da devam ediyor.

Root:

```text
/nfs/production/arl/chembl/tevfik/DEEP_APBS_DATASETS/runs/work12_pdbbind_box36_span70_v1_ablation_resnet3d4l_150epoch_thr040_lightval30_earlystop25
```

Amac: PDBbind box36/span70 uzerinde feature ablation yapmak.

Ana feature aileleri:

- `apbs_only`
- `shape_only`
- `apbs_shape`
- `shape_selected_chem`
- `apbs_shape_selected_chem`
- APBS representation kontrolleri

Su ana kadar ana bilimsel sinyal:

- APBS tek basina final model degil.
- APBS + shape + selected chemistry en guclu ailelerden biri.
- `shape_only` ve `shape_selected_chem` guclu baseline olduklari icin APBS katkisinin bu baseline'lara karsi raporlanmasi gerekiyor.

## Work 13 - PDBbind box72/span120 SelectedChem 5-fold

Durum: Codon'da tamamlandi, post-hoc re-evaluation bekliyor.

Root:

```text
/nfs/production/arl/chembl/tevfik/DEEP_APBS_DATASETS/runs/work13_pdbbind_box72_span120_v1_selectedchem_resnet3d4l_lr2e4_5fold_150epoch_thr040_lightval30
```

Amac: box36'dan box72/span120'ye cikinca selectedchem performansi artiyor mu sorusunu cevaplamak.

Ilk goruntu:

- box72 5 fold tamamlandi.
- Ilk skorlar box36'dan net daha iyi gorunmuyor.
- Kesin karar yeni DCC/PLI re-evaluation sonrasinda verilmeli.

## Work 14 - PDBbind New-Metric Re-evaluation

Durum: Yeni eklendi / current jobs bittikten sonra yapilacak.

Amac: Work10/12/13 checkpointlerini ayni checkpointlerle, yeni metrik koduyla tekrar degerlendirmek.

Gereken metrikler:

- DCC using `label_center`
- `dcc_to_ligand_angstrom`
- DCA
- DVO(all)
- DVO(DCC-success)
- PLI(all)
- PLI(DCC-success)
- Pocket-F1
- Top-k variants where applicable

Cikti politikasi:

- Eski log/CSV dosyalari overwrite edilmeyecek.
- Yeni raporlar ayri klasore yazilacak: `reeval_new_dcc_pli_YYYYMMDD`.

## Work 15 - scPDB Kalasanty 10-fold ResNet3D4L Benchmark

Durum: Yeni eklendi / current jobs bittikten sonra baslatilacak.

Amac: Kalasanty'nin scPDB 10-fold splitlerini aynen kullanarak bizim APBS-aware modelimizi karsilastirmak.

Split kurali:

- `generate_cache/data/kalasanty/train_ids_fold0..9` ve `test_ids_fold0..9` aynen kullanilir.
- Yalnizca ilgili ID icin H5 varsa split'e dahil edilir.
- H5 yoksa case raporda missing olarak kalir, fold assignment bozulmaz.

Default model:

```text
ResNet3D4L
```

Default feature setler:

```text
apbs_v1_full_signed_shape
apbs_v1_full_signed_shape_selected_chem
apbs_v1_full_signed_shape_selected_chem_surface
```

Default hyperparameter:

```text
learning_rate=1e-4
loss_alpha=0.5
pos_weight=2
weight_decay=1e-5
```

## Work 16 - scPDB PUResNet 5020 ResNet3D4L Benchmark

Durum: Yeni eklendi / current jobs bittikten sonra baslatilacak.

Amac: PUResNet-v1 final scPDB 5020 subseti uzerinde bizim APBS-aware modelimizi degerlendirmek.

Split durumu:

- Public PUResNet repo exact fold assignment vermiyor.
- Bu yuzden available-H5 subset uzerinden deterministic 4-fold uretilir.
- Rapor adi: "PUResNet 5020 available-H5 deterministic 4-fold", "exact paper folds" degil.

Default model ve feature setler Work15 ile ayni tutulur.

## Work 17 - scPDB UNetPlusPlus3D Follow-up Benchmark

Durum: Yeni eklendi / kullanici karariyla plana alindi.

Amac: Work8'de en guclu model ailesi olan `UNetPlusPlus3D`'yi, Kalasanty ve PUResNet scPDB benchmarklarinda ResNet3D4L baseline'dan sonra test etmek.

Neden gerekli:

- Work8 combined feature sweep'te `UNetPlusPlus3D`, `ResNet3D4L` ve `UNet3D4LA`'yi ortalamada gecti.
- En iyi Work8 kosusu `UNetPlusPlus3D + apbs_shape + full_signed` idi.
- Literature benchmarkta sadece ResNet3D4L kullanmak hizli ve kontrollu, ama final model adayi icin UNet++ mutlaka kontrol edilmeli.

Kapsam:

- Once Kalasanty 10-fold'da en iyi 1-2 feature set.
- Sonra PUResNet deterministic 4-fold'da ayni feature setler.
- ResNet baseline ile ayni split ve ayni metric protokolu kullanilir.

Onerilen feature setler:

```text
apbs_v1_full_signed_shape
apbs_v1_full_signed_shape_selected_chem
```

Egitim notu:

- Work8'de en iyi UNet++ epochlari 179-250 araliginda geldi.
- Bu yuzden UNet++ icin 150 epoch erken kalabilir; kaynak uygunsa 200 veya 250 epoch + early stopping daha mantikli.
- Ilk kosuda hyperparameter baseline, ResNet ile ayni tutulabilir. Sonra Work18 ile UNet++ icin ayrica tune edilir.

## Work 18 - UNetPlusPlus3D Loss and Capacity Mini-sweep

Durum: Yeni eklendi / Work17 sonrasi kontrollu follow-up.

Amac: UNet++ icin loss, augmentation ve kapasite ayarlarini kucuk bir matrisle test etmek.

Denenecekler:

- `BCEDiceLoss`, `alpha=0.5`, `pos_weight=2` baseline.
- `BCEFocalTverskyLoss`, FP-heavy variant: `alpha_fp=0.7`, `beta_fn=0.3`.
- `BCEFocalTverskyLoss`, recall/PLI-friendly variant: `alpha_fp=0.3`, `beta_fn=0.7`.
- `base_features=8` ve gerekirse `base_features=12`.
- rotate-only augmentation kontrolu.

Kural:

- Bu ana benchmark matrisi degil, mini-sweep olmalidir.
- Sadece en iyi fold/feature uzerinde baslatilir.
- PLI ve DVO(success) iyilesmeden daha buyuk matrise tasinmaz.

## Work 19 - SwinSiteLike3D Modern Model Pilot

Durum: Yeni model onerisi / opsiyonel pilot.

Amac: Transformer bottleneck iceren modern bir dense grid adayinin, ResNet3D4L ve UNetPlusPlus3D'ye karsi sinyal verip vermedigini kontrol etmek.

Model:

```text
SwinSiteLike3D
```

Neden bu model:

- `SwinSiteLike3D` zaten `models/LiteratureModels3D.py` icinde implemente ve `main.py` icinde kayitli.
- Mevcut dense H5 dataloader ile calisabilir.
- Exact SwinSite degildir; dense CNN + transformer bottleneck modern adayidir.
- Yeni sparse tensor pipeline veya yeni dependency gerektirmez.

Onerilen pilot:

- Tek dataset: Kalasanty fold0 veya PUResNet fold0.
- Tek feature set: `apbs_v1_full_signed_shape`.
- Karsilastirma: ayni fold'daki `ResNet3D4L` ve `UNetPlusPlus3D`.
- Basarili sayilma sarti: Pocket-F1/DCC benzer seviyeye gelirken DVO(success) veya PLI(success) artmali.

Kural:

- Ilk pilot zayifsa tam 10-fold/4-fold matrise tasinmaz.

## Work 20 - Literature-like Architecture Controls

Durum: Planlandi / opsiyonel.

Kaynak not:

```text
3dunet_configurable/reports/literature_models_and_topk_metrics_plan_2026-05-14.md
```

Amac: Tez savunmasinda "Kalasanty/PUResNet mimarisiyle denedin mi?" sorusuna kontrollu cevap vermek.

Modeller:

- `KalasantyUNet3D`
- `PUResNetV1Like3D`
- `PUResNetV2DenseLike3D`

Not:

- `PUResNetV2DenseLike3D`, exact sparse PUResNetV2 degildir.
- Exact PUResNetV2 icin MinkowskiEngine/sparse tensor pipeline gerekir; bu tez ana hattina alinmayacaksa dense proxy olarak raporlanmali.

## Work 21 - External Benchmark Preparation and Evaluation

Durum: Planlandi.

Amac: BU48, COACH420 ve PDBbind external/held-out split uzerinde final adaylari test etmek.

Kural:

- Threshold external test uzerinde optimize edilmeyecek.
- Validation'dan secilen threshold/postprocess dis benchmarka aynen uygulanacak.
- Literature comparison icin DCC/DCA/DVO/Pocket-F1/PLI ayni protokolle raporlanacak.

## Work 22 - Threshold and Postprocess Calibration

Durum: Final adaylar secildikten sonra.

Amac: Egitimden bagimsiz postprocess/threshold secimini kalibre etmek.

Denenecekler:

- Threshold median/validation-selected threshold.
- Top-1 vs Top-3 component reporting.
- Minimum component volume.
- No-prediction handling.
- Component score sum vs max/mean alternatives.

Kural:

- Test setinde threshold optimize edilmeyecek.
- Calibration sadece validation split uzerinden yapilacak.

## Work 23 - Error Analysis and APBS Interpretation

Durum: Final benchmarklar sonrasi.

Amac: Basari/basarisizlik orneklerini bilimsel olarak yorumlamak.

Analiz eksenleri:

- APBS katkisi hangi proteinlerde artiyor?
- Shape-only neden bu kadar guclu?
- Kimyasal feature'lar DCA/DVO tarafinda ne zaman yardim ediyor?
- Label source `binding_site_cavity6` oldugunda hangi pocket tipleri zor?
- DCC success olup DVO dusuk olan ornekler.
- PLI dusuk ama DCC iyi olan ornekler.

## Work 24 - Prediction CLI and 3D Web Analysis Page

Durum: Tez sonucu stabilize olduktan sonra.

Amac: Final modelin protein-only prediction modunda kullanilabilir hale gelmesi.

Kapsam:

- Config-driven prediction CLI.
- H5/cache format audit.
- Protein-only inference.
- Pocket component export.
- 3D visualization page.

Kural:

- Tez ana sonuclari bitmeden bu is ana odak olmayacak.

## Model Karari

Su anki pratik model hiyerarsisi:

1. `ResNet3D4L`: ucuz, stabil, folded baseline icin iyi.
2. `UNetPlusPlus3D`: Work8 kazanan model ailesi; final aday kontrolu icin gerekli.
3. `SwinSiteLike3D`: tek modern transformer-style pilot olarak denenebilir.
4. `KalasantyUNet3D` / `PUResNetV1Like3D`: literature-control icin yararli, ana final model olmak zorunda degil.

## Net Sonraki Plan

Yeni uzun egitim baslatma sirasi:

1. Mevcut Codon job'lari bitsin.
2. PDBbind Work10/12/13 yeni DCC/PLI ile re-evaluate edilsin.
3. Kalasanty scPDB 10-fold ResNet3D4L baslatilsin.
4. PUResNet 5020 deterministic 4-fold ResNet3D4L baslatilsin.
5. Bu iki benchmarkin ResNet sonuclari goruldukten sonra `UNetPlusPlus3D` follow-up baslatilsin.
6. UNet++ umut verirse Work18 loss/capacity mini-sweep yapilsin.
7. Ek modern model olarak yalnizca `SwinSiteLike3D` pilot kosulsun; iyi cikmazsa buyutulmesin.
