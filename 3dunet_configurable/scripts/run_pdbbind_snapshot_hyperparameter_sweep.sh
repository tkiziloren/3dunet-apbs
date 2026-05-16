#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGURABLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/nfs/production/arl/chembl/tevfik/DEEP_APBS_DATASETS}"
H5_DIR="${H5_DIR:-$DATA_ROOT/cache/training_snapshots/pdbbind/refined-set/box36_span70_v1_snapshot_20260516}"
LABEL="${LABEL:-binding_site_in_dataset}"
FOLDS="${FOLDS:-0}"
SPLIT_COUNT="${SPLIT_COUNT:-5}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SPLIT_DIR="${SPLIT_DIR:-$H5_DIR/splits_cache_kfold${SPLIT_COUNT}_seed${SPLIT_SEED}}"

EPOCHS="${EPOCHS:-150}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
VALIDATION_THRESHOLD="${VALIDATION_THRESHOLD:-0.40}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-$BATCH_SIZE}"
MODEL="${MODEL:-ResNet3D4L}"
BASE_FEATURES="${BASE_FEATURES:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_CONFIG="${BASE_CONFIG:-$CONFIGURABLE_DIR/config/local/expanded93/gridfix_expanded93_oldbest_dataset_electrostatic_shape_compact_chem.yml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/runs/work10_pdbbind_box36_span70_v1_hyperparameter_${MODEL}_${EPOCHS}epoch_thr${VALIDATION_THRESHOLD/./}}"
CONFIG_DIR="${CONFIG_DIR:-$OUTPUT_ROOT/generated_configs}"
RUN_FILTER="${RUN_FILTER:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CLEAN_INCOMPLETE="${CLEAN_INCOMPLETE:-1}"
DRY_RUN="${DRY_RUN:-0}"

cd "$CONFIGURABLE_DIR"
mkdir -p "$OUTPUT_ROOT" "$CONFIG_DIR"

echo "PDBBind snapshot hyperparameter sweep"
echo "Repo: $CONFIGURABLE_DIR"
echo "H5 snapshot: $H5_DIR"
echo "Label: $LABEL"
echo "Split dir: $SPLIT_DIR"
echo "Folds: $FOLDS"
echo "Python: $PYTHON_BIN"
echo "Model: $MODEL"
echo "Base features: $BASE_FEATURES"
echo "Epochs: $EPOCHS"
echo "Fixed validation threshold: $VALIDATION_THRESHOLD"
echo "Batch size: $BATCH_SIZE"
echo "Validation batch size: $VALIDATION_BATCH_SIZE"
echo "Output root: $OUTPUT_ROOT"
echo "Generated configs: $CONFIG_DIR"
echo "Run filter: ${RUN_FILTER:-<all>}"
echo "Dry run: $DRY_RUN"

if [[ ! -f "$SPLIT_DIR/fold0_train_cases.txt" ]]; then
  echo "Split dir does not exist yet. Build it with scripts/run_pdbbind_snapshot_kfold_ablation.sh first."
  exit 1
fi

export BASE_CONFIG CONFIG_DIR EPOCHS EARLY_STOPPING_PATIENCE VALIDATION_THRESHOLD BATCH_SIZE VALIDATION_BATCH_SIZE RUN_FILTER SPLIT_DIR FOLDS H5_DIR LABEL
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
batch_size = int(os.environ["BATCH_SIZE"])
validation_batch_size = int(os.environ["VALIDATION_BATCH_SIZE"])
folds = [int(item.strip()) for item in os.environ["FOLDS"].split(",") if item.strip()]
run_filter = {item.strip() for item in os.environ.get("RUN_FILTER", "").split(",") if item.strip()}

feature_sets = OrderedDict(
    [
        (
            "apbs_v1_full_signed_only",
            ["electrostatic_grid_v1_ligand_proximal_chains_7A_full_signed150"],
        ),
        (
            "apbs_v1_full_signed_shape",
            ["electrostatic_grid_v1_ligand_proximal_chains_7A_full_signed150", "shape"],
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
    ]
)

hyperparams = [
    {"name": "base_lr1e4_alpha05_pos1", "lr": 1e-4, "alpha": 0.5, "pos_weight": 1.0, "weight_decay": 1e-5},
    {"name": "lr3e4_alpha05_pos1", "lr": 3e-4, "alpha": 0.5, "pos_weight": 1.0, "weight_decay": 1e-5},
    {"name": "lr5e5_alpha05_pos1", "lr": 5e-5, "alpha": 0.5, "pos_weight": 1.0, "weight_decay": 1e-5},
    {"name": "lr1e4_alpha05_pos2", "lr": 1e-4, "alpha": 0.5, "pos_weight": 2.0, "weight_decay": 1e-5},
    {"name": "lr1e4_alpha05_pos5", "lr": 1e-4, "alpha": 0.5, "pos_weight": 5.0, "weight_decay": 1e-5},
    {"name": "lr1e4_alpha07_pos1", "lr": 1e-4, "alpha": 0.7, "pos_weight": 1.0, "weight_decay": 1e-5},
    {"name": "lr1e4_alpha03_pos1", "lr": 1e-4, "alpha": 0.3, "pos_weight": 1.0, "weight_decay": 1e-5},
    {"name": "lr1e4_alpha07_pos2", "lr": 1e-4, "alpha": 0.7, "pos_weight": 2.0, "weight_decay": 1e-5},
]

with base_config_path.open() as handle:
    base_config = yaml.safe_load(handle)

config_dir.mkdir(parents=True, exist_ok=True)
config_paths = []
metadata_rows = []
threshold_sweep = sorted({0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, validation_threshold})

requested_runs = [
    (fold_idx, feature_suffix, features, hp)
    for fold_idx in folds
    for feature_suffix, features in feature_sets.items()
    for hp in hyperparams
    if not run_filter or feature_suffix in run_filter or hp["name"] in run_filter
]
total_run_count = len(requested_runs)

for global_run_index, (fold_idx, feature_suffix, features, hp) in enumerate(requested_runs, start=1):
    train_file = split_dir / f"fold{fold_idx}_train_cases.txt"
    validation_file = split_dir / f"fold{fold_idx}_validation_cases.txt"
    if not train_file.exists() or not validation_file.exists():
        raise SystemExit(f"Missing split files for fold {fold_idx} under {split_dir}")

    config = copy.deepcopy(base_config)
    config["name"] = f"pdbbind_box36_span70_v1_hparam_fold{fold_idx}_{feature_suffix}_{hp['name']}"
    config["h5_directory"] = h5_dir
    config["label"] = label
    config["features"] = features
    config["feature_set"] = {
        "name": f"fold{fold_idx}_{feature_suffix}_{hp['name']}",
        "feature_name": feature_suffix,
        "hyperparameter_name": hp["name"],
        "fold": fold_idx,
        "index": global_run_index,
        "count": total_run_count,
    }
    config["training"]["num_epochs"] = epochs
    config["training"]["early_stopping_patience"] = early_stopping_patience
    config["training"]["batch_size"] = batch_size
    config["training"]["learning_rate"] = float(hp["lr"])
    config["training"]["weight_decay"] = float(hp["weight_decay"])
    config["training"].setdefault("loss", {})
    config["training"]["loss"]["alpha"] = float(hp["alpha"])
    config["training"]["loss"]["pos_weight"] = float(hp["pos_weight"])
    config["training"]["loss"]["dynamic_pos_weight"] = False
    config["validation"]["batch_size"] = validation_batch_size
    config["validation"]["threshold"] = validation_threshold
    config["validation"]["threshold_sweep"] = threshold_sweep
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
            "feature_set_name": feature_suffix,
            "hyperparameter_name": hp["name"],
            "run_index": global_run_index,
            "run_count": total_run_count,
            "feature_count": len(features),
            "features": ",".join(features),
            "learning_rate": hp["lr"],
            "loss_alpha": hp["alpha"],
            "pos_weight": hp["pos_weight"],
            "weight_decay": hp["weight_decay"],
            "train_file": str(train_file),
            "validation_file": str(validation_file),
            "fixed_validation_threshold": validation_threshold,
            "epochs": epochs,
            "early_stopping_patience": early_stopping_patience,
            "batch_size": batch_size,
            "validation_batch_size": validation_batch_size,
        }
    )

list_path = config_dir / "config_list.txt"
list_path.write_text("\n".join(str(path) for path in config_paths) + "\n")

metadata_path = config_dir / "hyperparameter_sets.csv"
with metadata_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "run",
            "fold",
            "feature_set_name",
            "hyperparameter_name",
            "run_index",
            "run_count",
            "feature_count",
            "features",
            "learning_rate",
            "loss_alpha",
            "pos_weight",
            "weight_decay",
            "train_file",
            "validation_file",
            "fixed_validation_threshold",
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
print(f"Hyperparameter metadata: {metadata_path}")
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
echo "Summarizing PDBBind hyperparameter runs"
"$PYTHON_BIN" scripts/summarize_gridfix_runs.py \
  --output-root "$OUTPUT_ROOT" \
  --write-csv "$OUTPUT_ROOT/run_summary.csv" | tee "$OUTPUT_ROOT/summary_stdout.txt"

echo "Done."
echo "Summary CSV: $OUTPUT_ROOT/run_summary.csv"
