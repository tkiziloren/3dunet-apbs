#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGURABLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/nfs/production/arl/chembl/tevfik/DEEP_APBS_DATASETS}"
H5_DIR="${H5_DIR:-$DATA_ROOT/cache/training_snapshots/pdbbind/refined-set/box36_span70_v1_snapshot_20260516}"
LABEL="${LABEL:-binding_site_in_dataset}"
FOLDS="${FOLDS:-0,1,2,3,4}"
SPLIT_COUNT="${SPLIT_COUNT:-5}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SPLIT_DIR="${SPLIT_DIR:-$H5_DIR/splits_cache_kfold${SPLIT_COUNT}_seed${SPLIT_SEED}}"
MANIFEST="${MANIFEST:-$H5_DIR/manifest.csv}"

EPOCHS="${EPOCHS:-150}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-25}"
VALIDATION_THRESHOLD="${VALIDATION_THRESHOLD:-0.40}"
PAPER_METRICS_START_EPOCH="${PAPER_METRICS_START_EPOCH:-31}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-$BATCH_SIZE}"
MODEL="${MODEL:-ResNet3D4L}"
BASE_FEATURES="${BASE_FEATURES:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_CONFIG="${BASE_CONFIG:-$CONFIGURABLE_DIR/config/local/expanded93/gridfix_expanded93_oldbest_dataset_electrostatic_shape_compact_chem.yml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/runs/work12_pdbbind_box36_span70_v1_ablation_${MODEL}_${EPOCHS}epoch_thr${VALIDATION_THRESHOLD/./}}"
CONFIG_DIR="${CONFIG_DIR:-$OUTPUT_ROOT/generated_configs}"
RUN_FILTER="${RUN_FILTER:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CLEAN_INCOMPLETE="${CLEAN_INCOMPLETE:-1}"
DRY_RUN="${DRY_RUN:-0}"

cd "$CONFIGURABLE_DIR"
mkdir -p "$OUTPUT_ROOT" "$CONFIG_DIR"

echo "PDBBind snapshot k-fold ablation"
echo "Repo: $CONFIGURABLE_DIR"
echo "H5 snapshot: $H5_DIR"
echo "Label: $LABEL"
echo "Split dir: $SPLIT_DIR"
echo "Folds: $FOLDS"
echo "Python: $PYTHON_BIN"
echo "Model: $MODEL"
echo "Base features: $BASE_FEATURES"
echo "Epochs: $EPOCHS"
echo "Early stopping patience: $EARLY_STOPPING_PATIENCE"
echo "Fixed validation threshold: $VALIDATION_THRESHOLD"
echo "Paper/top-k metrics start epoch: $PAPER_METRICS_START_EPOCH"
echo "Batch size: $BATCH_SIZE"
echo "Validation batch size: $VALIDATION_BATCH_SIZE"
echo "Output root: $OUTPUT_ROOT"
echo "Generated configs: $CONFIG_DIR"
echo "Run filter: ${RUN_FILTER:-<all>}"
echo "Dry run: $DRY_RUN"

required_features=(
  "shape"
  "electrostatic_grid_v1_ligand_proximal_chains_7A_clip20_minmax"
  "electrostatic_grid_v1_ligand_proximal_chains_7A_full_signed150"
  "electrostatic_grid_v1_ligand_proximal_chains_7A_positive_clip20"
  "electrostatic_grid_v1_ligand_proximal_chains_7A_negative_clip20"
  "atomic_donor"
  "atomic_acceptor"
  "atomic_hydrophobic"
  "atomic_aromatic"
  "dist_to_surface"
)

if [[ ! -d "$SPLIT_DIR" || ! -f "$SPLIT_DIR/fold0_train_cases.txt" ]]; then
  echo
  echo "Building k-fold splits under $SPLIT_DIR"
  split_args=(
    scripts/build_cache_kfold_splits.py
    --h5-dir "$H5_DIR"
    --output-dir "$SPLIT_DIR"
    --label "$LABEL"
    --folds "$SPLIT_COUNT"
    --seed "$SPLIT_SEED"
    --allow-label-atoms-outside-box
  )
  if [[ -f "$MANIFEST" ]]; then
    split_args+=(--manifest "$MANIFEST")
  fi
  for feature_name in "${required_features[@]}"; do
    split_args+=(--required-feature "$feature_name")
  done
  "$PYTHON_BIN" "${split_args[@]}"
else
  echo "Using existing split dir: $SPLIT_DIR"
fi

export BASE_CONFIG CONFIG_DIR EPOCHS EARLY_STOPPING_PATIENCE VALIDATION_THRESHOLD PAPER_METRICS_START_EPOCH BATCH_SIZE VALIDATION_BATCH_SIZE RUN_FILTER SPLIT_DIR FOLDS H5_DIR LABEL
"$PYTHON_BIN" - <<'PY'
import copy
import csv
import os
from collections import OrderedDict
from pathlib import Path

import yaml

base_config_path = Path(os.environ["BASE_CONFIG"])
config_dir = Path(os.environ["CONFIG_DIR"])
split_dir = Path(os.environ["SPLIT_DIR"])
h5_dir = os.environ["H5_DIR"]
label = os.environ["LABEL"]
epochs = int(os.environ["EPOCHS"])
early_stopping_patience = int(os.environ["EARLY_STOPPING_PATIENCE"])
validation_threshold = float(os.environ["VALIDATION_THRESHOLD"])
paper_metrics_start_epoch = int(os.environ["PAPER_METRICS_START_EPOCH"])
batch_size = int(os.environ["BATCH_SIZE"])
validation_batch_size = int(os.environ["VALIDATION_BATCH_SIZE"])
folds = [int(item.strip()) for item in os.environ["FOLDS"].split(",") if item.strip()]
run_filter = {item.strip() for item in os.environ.get("RUN_FILTER", "").split(",") if item.strip()}

feature_sets = OrderedDict(
    [
        ("shape_only", ["shape"]),
        (
            "apbs_v1_clip20_only",
            ["electrostatic_grid_v1_ligand_proximal_chains_7A_clip20_minmax"],
        ),
        (
            "apbs_v1_full_signed_only",
            ["electrostatic_grid_v1_ligand_proximal_chains_7A_full_signed150"],
        ),
        (
            "apbs_v1_posneg_only",
            [
                "electrostatic_grid_v1_ligand_proximal_chains_7A_positive_clip20",
                "electrostatic_grid_v1_ligand_proximal_chains_7A_negative_clip20",
            ],
        ),
        (
            "apbs_v1_full_signed_shape",
            ["electrostatic_grid_v1_ligand_proximal_chains_7A_full_signed150", "shape"],
        ),
        (
            "shape_selected_chem",
            ["shape", "atomic_donor", "atomic_acceptor", "atomic_hydrophobic", "atomic_aromatic"],
        ),
        (
            "apbs_v1_full_signed_shape_selected_chem",
            [
                "electrostatic_grid_v1_ligand_proximal_chains_7A_full_signed150",
                "shape",
                "atomic_donor",
                "atomic_acceptor",
                "atomic_hydrophobic",
                "atomic_aromatic",
            ],
        ),
        (
            "apbs_v1_full_signed_shape_selected_chem_surface",
            [
                "electrostatic_grid_v1_ligand_proximal_chains_7A_full_signed150",
                "shape",
                "atomic_donor",
                "atomic_acceptor",
                "atomic_hydrophobic",
                "atomic_aromatic",
                "dist_to_surface",
            ],
        ),
    ]
)

forbidden_features = {"ligand", "dist_to_ligand"}
with base_config_path.open() as handle:
    base_config = yaml.safe_load(handle)

config_dir.mkdir(parents=True, exist_ok=True)
config_paths = []
metadata_rows = []
threshold_sweep = sorted({0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, validation_threshold})

requested_runs = [
    (fold_idx, suffix, features)
    for fold_idx in folds
    for suffix, features in feature_sets.items()
    if not run_filter or suffix in run_filter
]
total_run_count = len(requested_runs)

for global_run_index, (fold_idx, suffix, features) in enumerate(requested_runs, start=1):
    leaked = forbidden_features.intersection(features)
    if leaked:
        raise SystemExit(f"Feature set {suffix} contains forbidden leakage features: {sorted(leaked)}")

    train_file = split_dir / f"fold{fold_idx}_train_cases.txt"
    validation_file = split_dir / f"fold{fold_idx}_validation_cases.txt"
    if not train_file.exists() or not validation_file.exists():
        raise SystemExit(f"Missing split files for fold {fold_idx} under {split_dir}")

    config = copy.deepcopy(base_config)
    config["name"] = f"pdbbind_box36_span70_v1_ablation_fold{fold_idx}_{suffix}"
    config["h5_directory"] = h5_dir
    config["label"] = label
    config["features"] = features
    config["feature_set"] = {
        "name": f"fold{fold_idx}_{suffix}",
        "feature_name": suffix,
        "fold": fold_idx,
        "index": global_run_index,
        "count": total_run_count,
    }
    config["training"]["num_epochs"] = epochs
    config["training"]["early_stopping_patience"] = early_stopping_patience
    config["training"]["batch_size"] = batch_size
    config["training"].setdefault("loss", {})
    config["training"]["loss"]["dynamic_pos_weight"] = False
    config["validation"]["batch_size"] = validation_batch_size
    config["validation"]["threshold"] = validation_threshold
    config["validation"]["threshold_sweep"] = threshold_sweep
    config["validation"].setdefault("paper_metrics", {})
    config["validation"]["paper_metrics"]["full_evaluation_start_epoch"] = paper_metrics_start_epoch
    config.setdefault("datasets", {})
    config["datasets"]["train_file"] = str(train_file)
    config["datasets"]["validation_file"] = str(validation_file)

    config_path = config_dir / f"{config['name']}.yml"
    with config_path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    config_paths.append(config_path)
    metadata_rows.append(
        {
            "run": config_path.stem,
            "fold": fold_idx,
            "feature_set_name": suffix,
            "run_index": global_run_index,
            "run_count": total_run_count,
            "feature_count": len(features),
            "features": ",".join(features),
            "train_file": str(train_file),
            "validation_file": str(validation_file),
            "fixed_validation_threshold": validation_threshold,
            "paper_metrics_start_epoch": paper_metrics_start_epoch,
            "epochs": epochs,
            "early_stopping_patience": early_stopping_patience,
            "batch_size": batch_size,
            "validation_batch_size": validation_batch_size,
        }
    )

list_path = config_dir / "config_list.txt"
list_path.write_text("\n".join(str(path) for path in config_paths) + "\n")

metadata_path = config_dir / "feature_sets.csv"
with metadata_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "run",
            "fold",
            "feature_set_name",
            "run_index",
            "run_count",
            "feature_count",
            "features",
            "train_file",
            "validation_file",
            "fixed_validation_threshold",
            "paper_metrics_start_epoch",
            "epochs",
            "early_stopping_patience",
            "batch_size",
            "validation_batch_size",
        ],
    )
    writer.writeheader()
    writer.writerows(metadata_rows)

print(f"Wrote {len(config_paths)} configs")
print(f"Config list: {list_path}")
print(f"Feature metadata: {metadata_path}")
PY

CONFIGS=()
while IFS= read -r config_path; do
  [[ -n "$config_path" ]] && CONFIGS+=("$config_path")
done < "$CONFIG_DIR/config_list.txt"

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "Dry run enabled. Generated configs only; no training was started."
  printf '  %s\n' "${CONFIGS[@]}"
  exit 0
fi

for config_path in "${CONFIGS[@]}"; do
  run_name="$(basename "$config_path" .yml)"
  run_dir="$OUTPUT_ROOT/$run_name"
  log_path="$run_dir/log/training.log"
  final_model_path="$run_dir/weights/${MODEL}_final_model.pth"

  echo
  echo "============================================================"
  echo "Running $run_name"
  echo "============================================================"
  echo "Config: $config_path"
  echo "Training log: $log_path"
  echo "Tail with: tail -f $log_path"

  if [[ "$SKIP_COMPLETED" == "1" && -f "$final_model_path" ]]; then
    echo "Skipping completed run: $run_name"
    continue
  fi

  if [[ "$CLEAN_INCOMPLETE" == "1" && -d "$run_dir" && ! -f "$final_model_path" ]]; then
    echo "Cleaning incomplete run directory before restart: $run_dir"
    rm -rf "$run_dir"
  fi

  PYTORCH_ENABLE_MPS_FALLBACK=1 "$PYTHON_BIN" main.py \
    --config "$config_path" \
    --model "$MODEL" \
    --base_features "$BASE_FEATURES" \
    --num_workers "$NUM_WORKERS" \
    --base_model_output_dir "$OUTPUT_ROOT"
done

echo
echo "Summarizing PDBBind ablation runs"
"$PYTHON_BIN" scripts/summarize_gridfix_runs.py \
  --output-root "$OUTPUT_ROOT" \
  --write-csv "$OUTPUT_ROOT/run_summary.csv" | tee "$OUTPUT_ROOT/summary_stdout.txt"

echo "Done."
echo "Summary CSV: $OUTPUT_ROOT/run_summary.csv"
