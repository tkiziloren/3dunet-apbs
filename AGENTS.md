# AGENTS.md

## Purpose
This repository trains and evaluates 3D voxel-based neural networks for protein-ligand binding-site segmentation.

## Communication
- Answer the user in the language they use. Keep code, identifiers, comments, config keys, logs, and commit-style technical text in English.
- When the user asks for a specific action, do that action and stop there. Do not independently branch into extra checks, alternative next steps, or adjacent experiments unless they are strictly required to complete the requested action.
- If an additional action seems useful but is not required, propose it briefly and wait for the user's approval before doing it.
- If the requested action cannot be completed exactly, report that clearly and state the minimal viable next option without running it unless the user approves.

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
- For PUResNet-v1 scPDB comparisons, use the final 5020-complex scPDB subset, not the intermediate 5462 set. The public PUResNet repository provides the 5020 subset but does not expose the original fold assignment files; if deterministic folds are regenerated, report that they are not the exact paper folds.
- Kalasanty fold files under `generate_cache/data/kalasanty/` are scPDB 10-fold IDs, not PDBbind IDs.

## Training Best Practices
- Keep image and label transforms geometrically identical. For custom transforms, never rotate or flip the channel axis.
- Prefer MONAI dictionary transforms for paired 3D image/label augmentation.
- Compute validation F1 from global TP/FP/FN, and run threshold sweeps when comparing models.
- Report label sparsity, feature normalization, and voxel resolution with any F1 result.
- For resolution-fix caches, remember that `box72` is not 1 Angstrom per voxel; use the stored resolution for physical-distance metrics.
- Keep paper-style pocket metrics explicit: DCC should default to predicted pocket center versus actual binding-site label center; DCA is predicted center versus nearest ligand atom/mask voxel; DVO is predicted pocket mask versus actual binding-site label mask; PLI is predicted pocket mask coverage of the ligand mask. Report both all-protein and DCC-success subsets for DVO/PLI when comparing with PUResNet/Kalasanty.
- Do not run long training jobs or Slurm submissions unless the user explicitly asks.

## Verification
- For training-code changes, run a lightweight import/instantiate check for the touched models.
- For dataset changes, inspect at least one HDF5 sample and report feature keys, label keys, shapes, positive voxel ratios, and continuous feature ranges.
- For metric or loss changes, prefer a small synthetic tensor check before launching a large experiment.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.
The Graphify CLI is installed at `/Users/tevfik/Library/Python/3.14/bin/graphify`; if `graphify` is not on PATH, prepend `/Users/tevfik/Library/Python/3.14/bin` to PATH or call that absolute path.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Before answering architecture, dependency, module, service, controller, model, API, data-flow, or refactoring questions, first read `graphify-out/GRAPH_REPORT.md` if it exists.
- Use Graphify as a map of the codebase, but verify important details against the actual source files before changing code.
- Do not rely only on `graphify-out/graph.json` directly unless needed; prefer `GRAPH_REPORT.md`, Graphify query/path/explain output, and targeted source files.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- After making code changes that affect architecture, dependencies, imports, services, controllers, models, APIs, package structure, or important data flows, run `graphify update .` to refresh the graph. This is the working CLI refresh command for this installation; `graphify .` is not accepted by graphifyy 0.8.11.
- Do not run `graphify update .` after every tiny change. Only run it after meaningful structural changes.
- For small bug fixes, comments, formatting, or isolated one-line changes, do not refresh the graph unless the change affects dependencies or architecture.
- Before finalising a task after structural changes, check the updated `graphify-out/GRAPH_REPORT.md` and mention whether the graph was refreshed.
- If Graphify refresh fails, do not hide the failure. Report the error clearly and continue with source-code analysis.
