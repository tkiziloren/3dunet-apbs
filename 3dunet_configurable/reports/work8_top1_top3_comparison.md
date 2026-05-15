# Work8 Top-1 vs Top-3 Pocket Comparison

Date: 2026-05-14

Scope: completed Work8 runs with available `run_summary.csv` and per-protein paper metric CSV files.

## Important Interpretation

`Top-1` means the highest-scoring connected pocket component is evaluated.

`Top-3` means the prediction is counted as DCC-successful if any of the top 3 connected pocket components is within 4 A of the reference center.

Current CSVs store Top-3 DCC success, but they do not store the centers/masks of the second and third components. Therefore:

- Top-1 F1, DCC, DCA, DVO are directly available.
- Top-3 F1 can be recomputed from `top3_dcc_success_4a`.
- Top-3 DCC success rate is directly available.
- Top-3 DCA and Top-3 DVO cannot be computed exactly from the existing CSVs.
- The DCA and DVO values below are Top-1 component values unless explicitly stated otherwise.

## Overall Effect

Across 21 completed Work8 runs:

| Metric | Top-1 | Top-3 | Delta |
|---|---:|---:|---:|
| Pocket-F1 | 0.6802 | 0.7077 | +0.0274 |
| DCC@4A | 0.5159 | 0.5481 | +0.0322 |

Top-3 improves the reported localization success, but the improvement is modest. This suggests that most successful predictions are already in the top component, while a smaller number of proteins have the correct pocket as the second or third component.

## Best Runs by Top-1 Selection

| Rank | Run | Top-1 F1 | Top-3 F1 | Delta F1 | Top-1 DCC | Top-3 DCC | Delta DCC | Top-1 DCA | Top-1 DVO_success |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `UNetPlusPlus3D / apbs_shape / full_signed` | 0.7143 | 0.7294 | +0.0151 | 0.5556 | 0.5741 | +0.0185 | 0.7315 | 0.5354 |
| 2 | `UNetPlusPlus3D / apbs_shape_selected_chem / full_signed` | 0.6988 | 0.7066 | +0.0078 | 0.5370 | 0.5463 | +0.0093 | 0.7593 | 0.5513 |
| 3 | `UNetPlusPlus3D / apbs_shape_selected_chem / posneg_clip20` | 0.6909 | 0.7219 | +0.0310 | 0.5278 | 0.5648 | +0.0370 | 0.7685 | 0.5387 |
| 4 | `UNetPlusPlus3D / apbs_shape / clip20_minmax` | 0.6909 | 0.7143 | +0.0234 | 0.5278 | 0.5556 | +0.0278 | 0.7500 | 0.5325 |
| 5 | `CBAMUNet3D / apbs_shape / full_signed` | 0.7143 | 0.7442 | +0.0299 | 0.5556 | 0.5926 | +0.0370 | 0.7315 | 0.4917 |

## Best Runs by Top-3 F1

| Rank | Run | Top-3 F1 | Top-1 F1 | Top-3 DCC | Top-1 DCC | Top-1 DCA | Top-1 DVO_success |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `CBAMUNet3D / apbs_shape / full_signed` | 0.7442 | 0.7143 | 0.5926 | 0.5556 | 0.7315 | 0.4917 |
| 2 | `ResNet3D4L / apbs_shape / posneg_clip20` | 0.7368 | 0.6988 | 0.5833 | 0.5370 | 0.6944 | 0.4914 |
| 3 | `ResNet3D4L / apbs_shape / full_signed` | 0.7368 | 0.7143 | 0.5833 | 0.5556 | 0.7315 | 0.5068 |
| 4 | `UNet3D4LA / apbs_shape / full_signed` | 0.7294 | 0.6988 | 0.5741 | 0.5370 | 0.7500 | 0.4994 |
| 5 | `UNetPlusPlus3D / apbs_shape / full_signed` | 0.7294 | 0.7143 | 0.5741 | 0.5556 | 0.7315 | 0.5354 |

## By Model

| Model | Runs | Top-1 F1 | Top-3 F1 | Delta F1 | Top-1 DCC | Top-3 DCC | Delta DCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `CBAMUNet3D` | 3 | 0.6906 | 0.7191 | +0.0285 | 0.5278 | 0.5617 | +0.0340 |
| `ResNet3D4L` | 6 | 0.6730 | 0.7008 | +0.0279 | 0.5077 | 0.5401 | +0.0324 |
| `UNet3D4LA` | 6 | 0.6731 | 0.7062 | +0.0330 | 0.5077 | 0.5463 | +0.0386 |
| `UNetPlusPlus3D` | 6 | 0.6894 | 0.7103 | +0.0209 | 0.5262 | 0.5509 | +0.0247 |

## Conclusion

Top-3 evaluation increases Pocket-F1 and DCC@4A, but it does not completely change the ranking. The strongest Top-1 model remains highly competitive under Top-3:

`UNetPlusPlus3D / apbs_shape / full_signed`

However, if the benchmark allows Top-3-style success, `CBAMUNet3D / apbs_shape / full_signed` currently has the best Top-3 F1 among completed runs.

For publication-quality reporting, Top-1 should remain the primary metric unless the compared method explicitly reports Top-3. Top-3 can be reported as an additional diagnostic metric.

## Next Fix Needed

To compare Top-3 DCA and Top-3 DVO exactly, the evaluation code should store component-level metrics for each of the top 3 pockets:

- component rank
- component center
- DCC
- DCA
- DVO
- voxel count
- probability score

Then we can report true Top-1 vs Top-3 F1, DCC, DCA, and DVO consistently.
