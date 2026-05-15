# Work8A Top-k Metric Re-evaluation

Evaluated runs: 30

This report evaluates completed Work8 checkpoints without retraining. Top-k means the metric is allowed to choose the best matching pocket among the first k connected prediction components.

Files:

- `topk_summary_by_threshold.csv`: every run, threshold, and Top-k protocol.
- `topk_best_by_run.csv`: best threshold per run and Top-k protocol.
- `topk_per_protein.csv`: per-protein Top-k metrics.
- `topk_component_metrics.csv`: component-level centers, DCC, DCA, and DVO.

## Top-3 Best Rows

| Model | Feature set | APBS variant | threshold | Pocket-F1 | DCC@4A | DCA@4A | DVO(success) | DVO(all best) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CBAMUNet3D | apbs_shape | apbs_full_signed | 0.40 | 0.7442 | 0.5926 | 0.7963 | 0.4854 | 0.4048 |
| ResNet3D4L | apbs_shape | apbs_posneg_clip20 | 0.05 | 0.7368 | 0.5833 | 0.7963 | 0.4388 | 0.3679 |
| ResNet3D4L | apbs_shape | apbs_full_signed | 0.10 | 0.7368 | 0.5833 | 0.8148 | 0.4937 | 0.3914 |
| UNet3D4LA | apbs_shape | apbs_full_signed | 0.45 | 0.7294 | 0.5741 | 0.7870 | 0.4879 | 0.4016 |
| UNetPlusPlus3D | apbs_shape | apbs_full_signed | 0.50 | 0.7294 | 0.5741 | 0.7963 | 0.5334 | 0.4334 |
| UNetPlusPlus3D | apbs_shape | apbs_clip20_minmax | 0.20 | 0.7219 | 0.5648 | 0.8056 | 0.5221 | 0.4300 |
| UNet3D4LA | apbs_shape_selected_chem | apbs_full_signed | 0.90 | 0.7219 | 0.5648 | 0.8241 | 0.4757 | 0.3875 |
| UNetPlusPlus3D | apbs_shape_selected_chem | apbs_posneg_clip20 | 0.20 | 0.7219 | 0.5648 | 0.8056 | 0.5191 | 0.4145 |
| UNet3D4LA | apbs_shape_selected_chem | apbs_posneg_clip20 | 0.55 | 0.7143 | 0.5556 | 0.8148 | 0.5123 | 0.4279 |
| CBAMUNet3D | apbs_shape_selected_chem | apbs_clip20_minmax | 0.55 | 0.7143 | 0.5556 | 0.8148 | 0.4657 | 0.3795 |

## Top-(n+2) Best Rows

For the current scPDB cache `n=1`, so Top-(n+2) is equivalent to Top-3 unless a future dataset provides multiple reference pockets per protein.

| Model | Feature set | APBS variant | threshold | Pocket-F1 | DCC@4A | DCA@4A | DVO(success) | DVO(all best) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CBAMUNet3D | apbs_shape | apbs_full_signed | 0.40 | 0.7442 | 0.5926 | 0.7963 | 0.4854 | 0.4048 |
| ResNet3D4L | apbs_shape | apbs_posneg_clip20 | 0.05 | 0.7368 | 0.5833 | 0.7963 | 0.4388 | 0.3679 |
| ResNet3D4L | apbs_shape | apbs_full_signed | 0.10 | 0.7368 | 0.5833 | 0.8148 | 0.4937 | 0.3914 |
| UNet3D4LA | apbs_shape | apbs_full_signed | 0.45 | 0.7294 | 0.5741 | 0.7870 | 0.4879 | 0.4016 |
| UNetPlusPlus3D | apbs_shape | apbs_full_signed | 0.50 | 0.7294 | 0.5741 | 0.7963 | 0.5334 | 0.4334 |
| UNetPlusPlus3D | apbs_shape | apbs_clip20_minmax | 0.20 | 0.7219 | 0.5648 | 0.8056 | 0.5221 | 0.4300 |
| UNet3D4LA | apbs_shape_selected_chem | apbs_full_signed | 0.90 | 0.7219 | 0.5648 | 0.8241 | 0.4757 | 0.3875 |
| UNetPlusPlus3D | apbs_shape_selected_chem | apbs_posneg_clip20 | 0.20 | 0.7219 | 0.5648 | 0.8056 | 0.5191 | 0.4145 |
| UNet3D4LA | apbs_shape_selected_chem | apbs_posneg_clip20 | 0.55 | 0.7143 | 0.5556 | 0.8148 | 0.5123 | 0.4279 |
| CBAMUNet3D | apbs_shape_selected_chem | apbs_clip20_minmax | 0.55 | 0.7143 | 0.5556 | 0.8148 | 0.4657 | 0.3795 |
