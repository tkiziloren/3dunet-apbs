# Literature Model Additions and Top-K Metric Plan

Date: 2026-05-14

## Scope

This note records the model and metric changes requested during the active Work8 run. The model classes were added as future candidates, but the metric CSV schema should be changed only after Work8 finishes so that the current sequential training run does not produce mixed-format outputs.

## Added Model Candidates

The following model classes were added in `models/LiteratureModels3D.py` and registered in `main.py`:

| Model class | Purpose | Exact literature reproduction? | Notes |
|---|---|---:|---|
| `PUResNetV1Like3D` | Dense PyTorch implementation of PUResNet v1-like residual encoder-decoder | Close topology match | Uses PUResNet v1-style bottleneck blocks `1x1 -> 3x3 -> 1x1`, downsampling schedule `1,2,2,3,3`, and skip concatenation. Use `BASE_FEATURES=18` for closest channel count to the original. |
| `PUResNetV2DenseLike3D` | Dense proxy inspired by PUResNetV2 sparse residual encoder-decoder | No | PUResNetV2 is a sparse MinkowskiEngine model. This class is only a dense approximation usable with the current H5 voxel dataloader. Exact PUResNetV2 requires a sparse tensor dataset/training pipeline. |
| `KalasantyUNet3D` | Kalasanty-style 3D U-Net baseline | Close topology match | Uses 9 convolution blocks and pooling schedule `2,2,3,3`, matching the reported 36-grid bottleneck behavior. |
| `SwinSiteLike3D` | Hybrid CNN + transformer U-Net candidate inspired by SwinSite | No | Uses dense CNN encoder-decoder plus transformer bottleneck. Exact SwinSite uses hierarchical Swin Transformer blocks and its own 96-grid feature protocol. |

## Why Exact PUResNetV2 Is Separate

PUResNetV2 represents protein atoms as Minkowski sparse tensors, not dense voxel grids. Its prediction is atom/sparse-coordinate based and postprocessed with DBSCAN. Therefore an exact PUResNetV2 implementation is not just another `nn.Module` for the current dense H5 grid dataloader. It needs:

- sparse coordinate/features dataset,
- sparse collate function,
- MinkowskiEngine dependency,
- atom-level labels,
- DBSCAN pocket extraction,
- separate evaluation bridge to DCC/DCA/DVO.

## Metric CSV Extension To Add After Work8

Current code stores Top-3 DCC success but does not store exact component-level Top-3 DCC/DCA/DVO/F1. After Work8 finishes, extend `utils/pocket_metrics.py` and `main.py` CSV fields.

### Summary CSV fields

Add these fields to `POCKET_SUMMARY_FIELDNAMES`:

- `top1_pocket_f1`
- `top1_pocket_precision`
- `top1_pocket_recall`
- `top1_dcc_success_rate_4a`
- `top1_dca_success_rate_4a`
- `top1_mean_dvo_all`
- `top1_mean_dvo_dcc_success`
- `top3_pocket_f1`
- `top3_pocket_precision`
- `top3_pocket_recall`
- `top3_dcc_success_rate_4a`
- `top3_dca_success_rate_4a`
- `top3_mean_dvo_all`
- `top3_mean_dvo_dcc_success`
- `topnplus2_pocket_f1`
- `topnplus2_pocket_precision`
- `topnplus2_pocket_recall`
- `topnplus2_dcc_success_rate_4a`
- `topnplus2_dca_success_rate_4a`
- `topnplus2_mean_dvo_all`
- `topnplus2_mean_dvo_dcc_success`

### Per-protein CSV fields

For each evaluated protein and threshold, store component-level ranks:

- `top1_component_id`
- `top1_component_voxels`
- `top1_component_score_sum`
- `top1_dcc_angstrom`
- `top1_dca_angstrom`
- `top1_dvo`
- `top1_dcc_success_4a`
- `top1_dca_success_4a`
- same fields for `top2`, `top3`
- `topnplus2_k`
- `topnplus2_best_rank_by_dcc`
- `topnplus2_best_dcc_angstrom`
- `topnplus2_best_dca_angstrom`
- `topnplus2_best_dvo`
- `topnplus2_dcc_success_4a`
- `topnplus2_dca_success_4a`

## Important Definition

For DCC/DCA Top-K success:

```text
success = at least one of the first K predicted pockets is within 4 A
```

For Top-K DVO:

```text
use the DVO of the best DCC-successful component among the first K predictions;
if no DCC-successful component exists, use 0.
```

For Top-K Pocket-F1:

```text
TP = protein has at least one successful pocket within first K predictions
FP = protein has predictions but none successful within first K predictions
FN = protein has no predictions
```

This makes Top-3 and Top-(n+2) comparable to the DCC/DCA success logic used in SwinSite-style evaluation while keeping Top-1 as the stricter deployment-like score.

## Implementation Timing

Do not change active Work8 metric CSV code while Work8 is still running. Apply the CSV extension after Work8 completes, then re-evaluate best checkpoints to produce exact Top-1, Top-3, and Top-(n+2) DCC/DCA/DVO/F1 tables without retraining.
