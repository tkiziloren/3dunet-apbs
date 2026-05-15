# AGENTS.md

## Purpose
This repository trains and evaluates 3D voxel-based neural networks for protein-ligand binding-site segmentation.

## Communication
- Answer the user in the language they use. Keep code, identifiers, comments, config keys, logs, and commit-style technical text in English.

## Active Code Path
- Treat `3dunet_configurable/` as the active implementation for new work.
- Treat `3dunet/` as the older parallel implementation unless the user explicitly targets it.
- Use configs under `3dunet_configurable/config/` to control features, labels, model settings, and train/validation/test splits.

## Project Direction
- Current priority is to finish the planned experimental work packages and lock defensible thesis results before doing framework/product work.
- After the experiment results are stable, turn the project into a protein binding-site prediction framework inspired by `pytorch-3dunet`: config-driven training, prediction, evaluation, standardized HDF5/cache format, pretrained checkpoints, and visualization.
- The thesis does not have to depend on the framework being complete. Treat the framework as a follow-up deliverable and CV-strengthening artifact unless the user explicitly changes priorities.
- Even before the framework phase, write new scripts and outputs in a framework-compatible style: clear CLI arguments, reproducible configs, stable output folders, machine-readable CSV/JSON summaries, and no hidden local assumptions.
- Keep the scientific positioning focused on APBS/electrostatics-aware 3D protein binding-site segmentation, standardized DCC/DCA/DVO/Pocket-F1 evaluation, and controlled feature/model ablations.

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
