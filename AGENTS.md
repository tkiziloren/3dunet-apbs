# AGENTS.md

## Purpose
This repository trains and evaluates 3D voxel-based neural networks for protein-ligand binding-site segmentation.

## Active Code Path
- Treat `3dunet_configurable/` as the active implementation for new work.
- Treat `3dunet/` as the older parallel implementation unless the user explicitly targets it.
- Use configs under `3dunet_configurable/config/` to control features, labels, model settings, and train/validation/test splits.

## Experiment Discipline
- Do not commit generated artifacts: TensorBoard event files, model checkpoints, prediction arrays, local run configs, logs, or cache files.
- Keep source/config changes separate from experiment-output cleanup.
- When changing a model, loss, transform, metric, or dataset contract, make the change reproducible through config or a clearly named script.
- Avoid using ligand-derived inputs for de novo binding-site prediction unless the experiment is explicitly ligand-conditioned.

## Training Best Practices
- Keep image and label transforms geometrically identical. For custom transforms, never rotate or flip the channel axis.
- Prefer MONAI dictionary transforms for paired 3D image/label augmentation.
- Compute validation F1 from global TP/FP/FN, and run threshold sweeps when comparing models.
- Report label sparsity, feature normalization, and voxel resolution with any F1 result.
- For resolution-fix caches, remember that `box72` is not 1 Angstrom per voxel; use the stored resolution for physical-distance metrics.
- Do not run long training jobs or Slurm submissions unless the user explicitly asks.

## Verification
- For training-code changes, run a lightweight import/instantiate check for the touched models.
- For dataset changes, inspect at least one HDF5 sample and report feature keys, label keys, shapes, positive voxel ratios, and continuous feature ranges.
- For metric or loss changes, prefer a small synthetic tensor check before launching a large experiment.
