# Work5 Model Sweep Report

## Scope

- Feature set: `apbs_only`
- APBS representation: `clip20`
- Fold: `fold1`
- Epochs: `250`
- Early stopping: disabled
- Fixed validation threshold: `0.40`
- Ranking key: validation selection score, then Pocket-F1, then voxel-F1

## Completed Models

Completed model count: `12`

| Rank | Model | Best epoch | Best threshold | Selection | Pocket-F1 | DCC@4A | DCA@4A | DVO success | Voxel-F1 | Fixed F1@0.40 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `ResNet3D4L` | 222 | 0.5 | 1.6599 | 0.6415 | 0.4722 | 0.7037 | 0.4888 | 0.5145 | 0.5143 |
| 2 | `UNet3D4LA` | 198 | 0.7 | 1.5675 | 0.6154 | 0.4444 | 0.7130 | 0.4449 | 0.4785 | 0.4762 |
| 3 | `UNetPlusPlus3D` | 202 | 0.3 | 1.5622 | 0.6154 | 0.4444 | 0.6667 | 0.4870 | 0.4730 | 0.4727 |
| 4 | `CBAMUNet3D` | 83 | 0.45 | 1.5599 | 0.6154 | 0.4444 | 0.6852 | 0.4831 | 0.4727 | 0.4724 |
| 5 | `ResidualUNet3D` | 69 | 0.45 | 1.5475 | 0.6154 | 0.4444 | 0.6852 | 0.4538 | 0.4633 | 0.4626 |
| 6 | `ResNet3D5L` | 238 | 0.2 | 1.5356 | 0.6154 | 0.4444 | 0.6667 | 0.4572 | 0.4711 | 0.4711 |
| 7 | `UNet3D4LStrided` | 247 | 0.5 | 1.5339 | 0.5974 | 0.4259 | 0.6944 | 0.4640 | 0.4593 | 0.4585 |
| 8 | `UNet3D4LAStrided` | 233 | 0.7 | 1.5299 | 0.6065 | 0.4352 | 0.6574 | 0.4749 | 0.4613 | 0.4601 |
| 9 | `UNet3D5L` | 176 | 0.9 | 1.5139 | 0.6329 | 0.4630 | 0.6852 | 0.4359 | 0.4620 | 0.4586 |
| 10 | `UNet3D4L` | 229 | 0.35 | 1.5076 | 0.6154 | 0.4444 | 0.6667 | 0.4328 | 0.4537 | 0.4532 |
| 11 | `SEResUNet3D` | 81 | 0.7 | 1.4394 | 0.5503 | 0.3796 | 0.6296 | 0.5038 | 0.4588 | 0.4570 |
| 12 | `LightweightUNet3D` | 41 | 0.35 | 1.3693 | 0.5695 | 0.3981 | 0.6296 | 0.3899 | 0.4110 | 0.4103 |

## Best By Metric

- Selection score: `ResNet3D4L` = `1.6599` (selection `1.6599`, Pocket-F1 `0.6415`, threshold `0.5`, epoch `222`)
- Pocket-F1: `ResNet3D4L` = `0.6415` (selection `1.6599`, Pocket-F1 `0.6415`, threshold `0.5`, epoch `222`)
- DCC@4A: `ResNet3D4L` = `0.4722` (selection `1.6599`, Pocket-F1 `0.6415`, threshold `0.5`, epoch `222`)
- DCA@4A: `UNet3D4LA` = `0.7130` (selection `1.5675`, Pocket-F1 `0.6154`, threshold `0.7`, epoch `198`)
- DVO success: `SEResUNet3D` = `0.5038` (selection `1.4394`, Pocket-F1 `0.5503`, threshold `0.7`, epoch `81`)
- Voxel-F1: `ResNet3D4L` = `0.5145` (selection `1.6599`, Pocket-F1 `0.6415`, threshold `0.5`, epoch `222`)
- Fixed F1@0.40: `ResNet3D4L` = `0.5143` (selection `1.6599`, Pocket-F1 `0.6415`, threshold `0.5`, epoch `222`)

## Interpretation

The strongest completed Work5 model is `ResNet3D4L`. It has the best selection score, Pocket-F1, DCC@4A, voxel-F1, and fixed-threshold F1 among completed models.

The main practical conclusion is that APBS-only performance is not only a feature-representation problem; architecture matters as well. `ResNet3D4L` appears to extract substantially more useful signal from APBS-only `clip20` than the baseline U-Net family.

ConvNeXt-style models should be treated carefully. The original heavy `ConvNeXtUNet3D` was stopped because it was too slow and showed poor early learning. Work6 uses lighter, 3D-friendly ConvNeXt-style variants instead.

## Files

- Output root: `/Users/tevfik/Sandbox/github/PHD/runs/work5_apbs_only_clip20_model_sweep_fold1_250epoch_thr040`
- Per-model summaries: `<output-root>/<model>/run_summary.csv`
