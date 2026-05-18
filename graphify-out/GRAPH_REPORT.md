# Graph Report - 3dunet-apbs  (2026-05-18)

## Corpus Check
- 153 files · ~258,531 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1258 nodes · 1750 edges · 125 communities (113 shown, 12 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 32 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2c166cb1`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 123|Community 123]]

## God Nodes (most connected - your core abstractions)
1. `Thesis Defense Question Bank - 2026-05-08` - 53 edges
2. `Next Work Packages - 2026-05-08` - 23 edges
3. `main()` - 18 edges
4. `main()` - 16 edges
5. `APBS Clipping and Ablation Explainer - 2026-05-08` - 15 edges
6. `Work8 Sonuç Raporu` - 13 edges
7. `evaluate_topk_metrics_for_sample()` - 12 edges
8. `validate_case()` - 12 edges
9. `evaluate_pocket_metrics_for_sample()` - 11 edges
10. `Work8 Design` - 11 edges

## Surprising Connections (you probably didn't know these)
- `NullLogger` --uses--> `Standardize`  [INFERRED]
  3dunet_configurable/scripts/visualize_pocket_predictions.py → 3dunet/transforms.py
- `get_loss_function()` --calls--> `BCEFocalTverskyLoss`  [INFERRED]
  3dunet/utils/training.py → 3dunet_configurable/utils/losses.py
- `Flattens a given tensor such that the channel axis is first.     The shapes are` --rationale_for--> `flatten()`  [EXTRACTED]
  3dunet_configurable/utils/losses.py → 3dunet/utils/losses.py
- `Loss fonksiyonu oluşturur.` --rationale_for--> `get_loss_function()`  [EXTRACTED]
  3dunet_configurable/utils/training.py → 3dunet/utils/training.py
- `F1, Precision, Recall ve Confusion Matrix metriklerini bir kez başlatır.      Ar` --rationale_for--> `initialize_metrics()`  [EXTRACTED]
  3dunet_configurable/utils/training.py → 3dunet/utils/training.py

## Communities (125 total, 12 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (53): append_csv_rows(), as_list(), build_monai_transforms(), build_transforms(), create_model(), format_count(), log_readable_validation_summary(), main() (+45 more)

### Community 1 - "Community 1"
Cohesion: 0.04
Nodes (53): 10. Neden graph kullanmadın?, 11. APBS neden bu iş için anlamlı bir feature?, 12. APBS feature'ında normalization neden bu kadar önemli?, 13. `dist2ligand` gibi ligand-derived feature'lar neden riskli?, 14. Binding-site label'ı nasıl tanımlanıyor? Dataset label ile calculated label farkı nedir?, 15. DCC, DCA, DVO, Pocket-F1 ve voxel-F1 arasındaki fark nedir?, 16. Modelin eski F1 skorları neden düşük çıkıyordu?, 17. Selection score nedir ve neden kullanıyoruz? (+45 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (43): _AbstractDiceLoss, BCEDiceLoss, BCEDiceLoss1, BCEDiceLoss2, BCEFocalTverskyLoss, compute_per_channel_dice(), DiceLoss, flatten() (+35 more)

### Community 3 - "Community 3"
Cohesion: 0.04
Nodes (46): BU48 ve COACH Hazırlığı, code:text (/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/3dunet-ap), code:text (/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/generate_), code:text (/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/generate_), code:text (/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/generate_), code:text (/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/3dunet-ap), code:text (ResNet3D4L x 11 feature set x fold1 x 250 epoch), code:text (/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/3dunet-ap) (+38 more)

### Community 4 - "Community 4"
Cohesion: 0.04
Nodes (45): 10 Gunluk Oncelik Sirası, 161:160 Downscale mi?, Benchmark/analysis mode, code:text (161 x 161 x 161 points), code:text (feature sets:), code:text (3 feature sets x 2 grid sizes x 1 fold x 8 configs = 48 trai), code:text (3 grid sizes x 5 feature sets x 5 folds), code:text (3 APBS feature sets x 2 grid sizes x 8 hyperparameter config) (+37 more)

### Community 5 - "Community 5"
Cohesion: 0.04
Nodes (44): 10.1 APBS Fiziksel Olarak Anlamlı, 10.2 APBS-only Sonuçları Bilimsel Olarak İlginç, 10.3 APBS'in DVO'ya Katkısı Özellikle Önemli, 10.4 Çalışma Sadece Model Değil, Representation Çalışması, 10. Neden Sunulmaya Değer?, 11. Hangi Koşullarda Yayın İçin Güçlü Olur?, 12. Şu Anki Zayıf Noktalar, 13. Benim Net Kanaatim (+36 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (18): _ConvNormAct3D, _DenseV2Decoder3D, _DenseV2ResidualBlock3D, _DenseV2Stage3D, _group_count(), _KalasantyDoubleConv3D, KalasantyUNet3D, _match_spatial_size() (+10 more)

### Community 7 - "Community 7"
Cohesion: 0.10
Nodes (23): load_config(), normalize_feature(), ProteinLigandDatasetWithH5, Normalize a feature using the default physical range or a config override., add_center(), add_scatter(), binary_stats(), choose_device() (+15 more)

### Community 8 - "Community 8"
Cohesion: 0.15
Nodes (11): ConvBlock, dice_coeff(), eval_epoch(), iou_score(), is_single_fragment(), main(), mol2_to_voxel(), Return True if mol2 file has exactly one disconnected fragment. (+3 more)

### Community 9 - "Community 9"
Cohesion: 0.10
Nodes (36): assign_balanced_folds(), discover_h5_cases(), find_dataset_path_in_any_group(), has_entry_suffix(), main(), normalize_id(), parse_args(), read_id_list() (+28 more)

### Community 10 - "Community 10"
Cohesion: 0.06
Nodes (30): All Chemical Features, Atomic Features, Baseline Denemeleri, Baseline - Sadece Electrostatic Grid, Chemical Features, code:bash (cd ~/PHD/3dunet-apbs/3dunet_configurable/slurm), code:bash (bash train_new.sh full_context_dataset config/codon_with_dif), code:bash (# UNet3D5L) (+22 more)

### Community 11 - "Community 11"
Cohesion: 0.13
Nodes (11): AttentionDecoder, AttentionGate, Decoder, DoubleConv, _group_count(), _match_spatial_size(), RepeatUpsample3D, SingleConv (+3 more)

### Community 12 - "Community 12"
Cohesion: 0.07
Nodes (27): APBS Temsil Biçimi Ortalamaları, code:text (5 model x 2 öznitelik grubu x 3 APBS temsil biçimi = 30 eğit), code:text (apbs_shape + apbs_full_signed), code:text (apbs_shape_selected_chem + apbs_full_signed), code:text (Tamamlanan eğitim: 30/30), code:text (UNetPlusPlus3D + apbs_shape + apbs_full_signed), code:text (selection score: 1.8376), code:text (UNetPlusPlus3D + apbs_shape_selected_chem + apbs_full_signed) (+19 more)

### Community 13 - "Community 13"
Cohesion: 0.07
Nodes (27): 3.1 `apbs_clip5_minmax`, 3.2 `apbs_clip10_minmax`, 3.3 `apbs_clip20_minmax`, 3.4 `apbs_no_cutoff_current`, 3.5 `apbs_full_minmax`, 3.6 `apbs_full_signed`, 3.7 `apbs_clip20_signed`, 3.8 `apbs_posneg_clip20` (+19 more)

### Community 14 - "Community 14"
Cohesion: 0.15
Nodes (8): AttentionBlock, Decoder, DecoderAttention, DoubleConv, Encoder, SingleConv, TripleConv, UNet3D4LAC

### Community 15 - "Community 15"
Cohesion: 0.15
Nodes (8): AttentionBlock, Decoder, DecoderAttention, DoubleConv, Encoder, SingleConv, TripleConv, UNet3D4LC

### Community 16 - "Community 16"
Cohesion: 0.20
Nodes (24): center(), default_skips(), default_unet_arrows(), Diagram, draw_arrow(), draw_legend(), draw_node(), draw_skip() (+16 more)

### Community 17 - "Community 17"
Cohesion: 0.08
Nodes (24): 10. Çalışma Şekli Önerisi, 1. Klasördeki Dosyalar, 2. Tez Önerisi Raporundan Çıkan Ana Hat, 3. Word Şablonundan Çıkan Tez Yapısı, 4. Tez Yazım Kılavuzundan Kritik Format Kuralları, 5. OU-09 Formundan Çıkan Süreç Notu, 6. Komite Profili İçin Tez Dilini Nasıl Ayarlamalıyız?, 7. Tez İçin Önerilen Güncel İçindekiler (+16 more)

### Community 18 - "Community 18"
Cohesion: 0.17
Nodes (7): AttentionBlock, Decoder, DecoderAttention, DoubleConv, Encoder, SingleConv, UNet3D4LA

### Community 19 - "Community 19"
Cohesion: 0.14
Nodes (11): AttentionBlock, ConvBlock, dice_coeff(), eval_epoch(), iou_score(), is_single_fragment(), main(), mol2_to_voxel() (+3 more)

### Community 20 - "Community 20"
Cohesion: 0.14
Nodes (11): AttentionBlock, ConvBlock, dice_coeff(), eval_epoch(), iou_score(), is_single_fragment(), main(), mol2_to_voxel() (+3 more)

### Community 21 - "Community 21"
Cohesion: 0.10
Nodes (20): Basari Kriteri, Beklenen Cevaplar, code:text (box36_span70), code:text (Top-1 Pocket-F1), code:text (legacy_apbs_only:), code:text (v2_apbs_full_raw_only:), code:text (ResNet3D4L), code:text (ResNet3D4LGN) (+12 more)

### Community 22 - "Community 22"
Cohesion: 0.08
Nodes (14): ConvBlock, dice_coeff(), eval_epoch(), iou_score(), is_single_fragment(), main(), mol2_to_voxel(), Return True if mol2 file has exactly one disconnected fragment. (+6 more)

### Community 23 - "Community 23"
Cohesion: 0.11
Nodes (8): CustomCompose, MonaiWrapper, RandomFlip, RandomRotate3D, Rastgele eksenlerde tensor üzerinde yansıma (flip) işlemi uygular., Basit bir Z-skoru normalizasyonu uygular., Tensor'a Z-skoru normalizasyonu uygular.          Args:             tensor (torc, Standardize

### Community 24 - "Community 24"
Cohesion: 0.11
Nodes (8): CustomCompose, MonaiWrapper, RandomFlip, RandomRotate3D, Rastgele eksenlerde tensor üzerinde yansıma (flip) işlemi uygular., Basit bir Z-skoru normalizasyonu uygular., Tensor'a Z-skoru normalizasyonu uygular.          Args:             tensor (torc, Standardize

### Community 25 - "Community 25"
Cohesion: 0.22
Nodes (5): Decoder, DoubleConv, Encoder, SingleConv, UNet3D4L

### Community 26 - "Community 26"
Cohesion: 0.22
Nodes (5): Decoder, DoubleConv, Encoder, SingleConv, UNet3D5L

### Community 27 - "Community 27"
Cohesion: 0.22
Nodes (5): Decoder, DoubleConv, Encoder, SingleConv, UNet3D6L

### Community 28 - "Community 28"
Cohesion: 0.21
Nodes (5): Decoder, DoubleConv, Encoder, SingleConv, UNet3D

### Community 29 - "Community 29"
Cohesion: 0.21
Nodes (5): Decoder, DoubleConv, Encoder, SingleConv, UNet3D

### Community 30 - "Community 30"
Cohesion: 0.13
Nodes (14): Amac, APBS Representation, Beklenen Cevap, Calistirma Komutu, Calistirma Scripti, code:text (5 model x 2 feature set x 3 APBS representation = 30 trainin), code:bash (cd /Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/3dunet), code:bash (tail -f /Users/tevfik/Sandbox/github/PHD/runs/work8_combined) (+6 more)

### Community 31 - "Community 31"
Cohesion: 0.26
Nodes (4): BasicBlock3D, DecoderResNet3D, EncoderResNet3D, ResNet3D4L

### Community 32 - "Community 32"
Cohesion: 0.26
Nodes (4): BasicBlock3D, DecoderResNet3D, EncoderResNet3D, ResNet3D5L

### Community 33 - "Community 33"
Cohesion: 0.26
Nodes (4): BasicBlock3D, DecoderResNet3D, EncoderResNet3D, ResNet3D6L

### Community 34 - "Community 34"
Cohesion: 0.19
Nodes (5): EfficientConvNeXtBlock3D, EfficientConvNeXtDecoder3D, EfficientConvNeXtStage3D, _group_count(), TinyConvNeXtUNet3D

### Community 35 - "Community 35"
Cohesion: 0.15
Nodes (12): Added Model Candidates, code:text (success = at least one of the first K predicted pockets is w), code:text (use the DVO of the best DCC-successful component among the f), code:text (TP = protein has at least one successful pocket within first), Implementation Timing, Important Definition, Literature Model Additions and Top-K Metric Plan, Metric CSV Extension To Add After Work8 (+4 more)

### Community 36 - "Community 36"
Cohesion: 0.17
Nodes (5): CBAMResNet3D4LGN, ResNet3D4LGN, _ResNet3D4LGNBase, ResNetGNEncoder3D, SEResNet3D4LGN

### Community 37 - "Community 37"
Cohesion: 0.36
Nodes (9): generate_cache_for_proteins(), generate_h5_file(), generate_label_h5_file(), generate_labels_h5_from_binding_site_in_parallel(), LabelMaskType, load_config(), parse_args(), pdb_to_grid() (+1 more)

### Community 38 - "Community 38"
Cohesion: 0.24
Nodes (3): ConvNeXtBlock3D, LightweightUNet3D, PlainConvBlock3D

### Community 39 - "Community 39"
Cohesion: 0.24
Nodes (5): CBAMUNet3D, ConvNeXtUNet3D, ResidualUNet3D, _ResidualUNet3DBase, SEResUNet3D

### Community 40 - "Community 40"
Cohesion: 0.22
Nodes (3): CBAM3D, ResNetGNBlock3D, SqueezeExcite3D

### Community 41 - "Community 41"
Cohesion: 0.27
Nodes (8): create_output_dirs(), load_config(), parse_args(), Config dosyasını yükler., Log, ağırlık ve TensorBoard dosyalarını kaydetmek için dizinleri oluşturur., Konsola ve dosyaya log yazan bir logger ayarlama fonksiyonu.      Args:, Argümanları okuyarak döner., setup_logger()

### Community 42 - "Community 42"
Cohesion: 0.20
Nodes (8): Active Code Path, Communication, Experiment Discipline, graphify, Project Direction, Purpose, Training Best Practices, Verification

### Community 43 - "Community 43"
Cohesion: 0.22
Nodes (8): Best Runs by Top-1 Selection, Best Runs by Top-3 F1, By Model, Conclusion, Important Interpretation, Next Fix Needed, Overall Effect, Work8 Top-1 vs Top-3 Pocket Comparison

### Community 44 - "Community 44"
Cohesion: 0.42
Nodes (8): as_float(), best_rows_for_run(), fixed_paper_f1_for_epoch(), load_best_threshold_rows(), main(), parse_args(), parse_config_from_log(), sort_key()

### Community 47 - "Community 47"
Cohesion: 0.29
Nodes (3): _match_spatial_size(), ResNetGNDecoder3D, UNetPlusPlus3D

### Community 48 - "Community 48"
Cohesion: 0.32
Nodes (3): ResidualBlock3D, ResidualDecoder3D, ResidualEncoder3D

### Community 49 - "Community 49"
Cohesion: 0.52
Nodes (5): generate_cache_for_proteins(), generate_h5_file(), load_config(), parse_args(), pdb_to_grid()

### Community 50 - "Community 50"
Cohesion: 0.29
Nodes (6): Best By Metric, Completed Models, Files, Interpretation, Scope, Work5 Model Sweep Report

### Community 51 - "Community 51"
Cohesion: 0.62
Nodes (6): as_float(), best_by(), fmt(), load_rows(), main(), parse_args()

### Community 54 - "Community 54"
Cohesion: 0.80
Nodes (3): main(), parse_log_file(), plot_metrics()

### Community 56 - "Community 56"
Cohesion: 0.83
Nodes (3): generate_configs(), load_yaml(), save_yaml()

### Community 57 - "Community 57"
Cohesion: 0.83
Nodes (3): load_cases(), main(), parse_args()

### Community 60 - "Community 60"
Cohesion: 0.83
Nodes (3): main(), parse_args(), select_cases()

### Community 61 - "Community 61"
Cohesion: 0.83
Nodes (3): main(), normalize_weight_name(), parse_args()

### Community 64 - "Community 64"
Cohesion: 0.50
Nodes (3): Top-3 Best Rows, Top-(n+2) Best Rows, Work8A Top-k Metric Re-evaluation

## Knowledge Gaps
- **272 isolated node(s):** `PreToolUse`, `Purpose`, `Communication`, `Active Code Path`, `Project Direction` (+267 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Community 0` to `Community 2`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Why does `NullLogger` connect `Community 7` to `Community 24`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **Why does `Standardize` connect `Community 24` to `Community 7`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `main()` (e.g. with `set_reproducibility()` and `calculate_pos_weight_from_loader()`) actually correct?**
  _`main()` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `main()` (e.g. with `create_model()` and `center_of_mask()`) actually correct?**
  _`main()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Rastgele eksenlerde tensor üzerinde yansıma (flip) işlemi uygular.`, `Basit bir Z-skoru normalizasyonu uygular.`, `Tensor'a Z-skoru normalizasyonu uygular.          Args:             tensor (torc` to the rest of the system?**
  _311 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.07792207792207792 - nodes in this community are weakly interconnected._