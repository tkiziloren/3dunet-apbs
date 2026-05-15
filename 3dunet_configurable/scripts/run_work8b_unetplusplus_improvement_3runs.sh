#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGURABLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$CONFIGURABLE_DIR/../.venv/bin/python}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/Users/tevfik/Sandbox/github/PHD/runs/work8b_unetplusplus_improvement_3runs_fold1_200epoch_thr040}"
CONFIG_DIR="${CONFIG_DIR:-$OUTPUT_ROOT/generated_configs}"
BASE_CONFIG="${BASE_CONFIG:-/Users/tevfik/Sandbox/github/PHD/runs/work8_combined_model_feature_representation_sweep_fold1_250epoch_thr040/UNetPlusPlus3D/apbs_shape/scpdb_apbs_cutoff_fold1_apbs_shape_apbs_full_signed/config_snapshot.yml}"
SPLIT_DIR="${SPLIT_DIR:-/Users/tevfik/Sandbox/github/PHD/data/scPDB_cache_gridfix_v1/label_cavity6/box36_span70/splits_cache_kfold5_seed42}"
FOLD="${FOLD:-1}"
EPOCHS="${EPOCHS:-200}"
VALIDATION_THRESHOLD="${VALIDATION_THRESHOLD:-0.40}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CLEAN_INCOMPLETE="${CLEAN_INCOMPLETE:-1}"
DRY_RUN="${DRY_RUN:-0}"

cd "$CONFIGURABLE_DIR"
mkdir -p "$OUTPUT_ROOT" "$CONFIG_DIR"

echo "Work8B UNetPlusPlus3D improvement sweep"
echo "Output root: $OUTPUT_ROOT"
echo "Generated configs: $CONFIG_DIR"
echo "Base config: $BASE_CONFIG"
echo "Fold: $FOLD"
echo "Epochs: $EPOCHS"
echo "Fixed validation threshold: $VALIDATION_THRESHOLD"
echo "Dry run: $DRY_RUN"

export BASE_CONFIG CONFIG_DIR SPLIT_DIR FOLD EPOCHS VALIDATION_THRESHOLD
"$PYTHON_BIN" - <<'PY'
import copy
import csv
import os
from pathlib import Path

import yaml

base_config_path = Path(os.environ["BASE_CONFIG"])
config_dir = Path(os.environ["CONFIG_DIR"])
split_dir = Path(os.environ["SPLIT_DIR"])
fold = int(os.environ["FOLD"])
epochs = int(os.environ["EPOCHS"])
validation_threshold = float(os.environ["VALIDATION_THRESHOLD"])

feature_sets = {
    "apbs_shape": ["electrostatic_grid", "shape"],
    "apbs_shape_selected_chem": [
        "electrostatic_grid",
        "shape",
        "atomic_donor",
        "atomic_acceptor",
        "atomic_hydrophobic",
        "atomic_aromatic",
    ],
}
full_signed_normalization = {
    "electrostatic_grid": {
        "min": -150.0,
        "max": 150.0,
        "clip": False,
        "normalize": True,
        "output_min": -1.0,
        "output_max": 1.0,
    }
}
threshold_sweep = sorted({0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, validation_threshold})

plan = [
    {
        "name": "work8b01_unetpp_apbs_shape_full_signed_channelwise",
        "description": "Best Work8 candidate with channel-wise standardization.",
        "model": "UNetPlusPlus3D",
        "base_features": 8,
        "feature_set": "apbs_shape",
        "flip_axis_prob": 0.5,
        "rotate90_prob": 1.0,
        "loss": {"type": "BCEDiceLoss", "alpha": 0.5, "smooth": 1.0, "pos_weight": 1, "dynamic_pos_weight": False},
    },
    {
        "name": "work8b02_unetpp_apbs_shape_full_signed_channelwise_rotate_only",
        "description": "Best Work8 candidate with channel-wise standardization and no flip augmentation.",
        "model": "UNetPlusPlus3D",
        "base_features": 8,
        "feature_set": "apbs_shape",
        "flip_axis_prob": 0.0,
        "rotate90_prob": 1.0,
        "loss": {"type": "BCEDiceLoss", "alpha": 0.5, "smooth": 1.0, "pos_weight": 1, "dynamic_pos_weight": False},
    },
    {
        "name": "work8b03_unetpp_apbs_shape_selected_chem_full_signed_focal_tversky_bf12",
        "description": "Best DVO-side candidate with channel-wise standardization, rotate-only augmentation, Focal-Tversky loss, and base_features=12.",
        "model": "UNetPlusPlus3D",
        "base_features": 12,
        "feature_set": "apbs_shape_selected_chem",
        "flip_axis_prob": 0.0,
        "rotate90_prob": 1.0,
        "loss": {
            "type": "BCEFocalTverskyLoss",
            "bce_weight": 0.5,
            "tversky_weight": 1.0,
            "alpha_fp": 0.7,
            "beta_fn": 0.3,
            "gamma": 1.33,
            "smooth": 1.0,
            "pos_weight": 1,
            "dynamic_pos_weight": False,
        },
    },
]

with base_config_path.open() as handle:
    base_config = yaml.safe_load(handle)

train_file = split_dir / f"fold{fold}_train_cases.txt"
validation_file = split_dir / f"fold{fold}_validation_cases.txt"
if not train_file.exists() or not validation_file.exists():
    raise SystemExit(f"Missing split files under {split_dir} for fold {fold}")

config_dir.mkdir(parents=True, exist_ok=True)
rows = []
for idx, item in enumerate(plan, start=1):
    config = copy.deepcopy(base_config)
    config["name"] = item["name"]
    config["features"] = feature_sets[item["feature_set"]]
    config["feature_normalization"] = full_signed_normalization
    config["feature_set"] = {
        "name": item["name"],
        "feature_name": item["feature_set"],
        "apbs_cutoff_variant": "apbs_full_signed",
        "fold": fold,
        "index": idx,
        "count": len(plan),
    }
    config.setdefault("metadata", {})
    config["metadata"]["work_package"] = "Work8B"
    config["metadata"]["experiment_description"] = item["description"]
    config["training"]["num_epochs"] = epochs
    config["training"]["early_stopping_patience"] = 0
    config["training"]["loss"] = item["loss"]
    config["validation"]["threshold"] = validation_threshold
    config["validation"]["threshold_sweep"] = threshold_sweep
    config["datasets"]["train_file"] = str(train_file)
    config["datasets"]["validation_file"] = str(validation_file)
    config.setdefault("augmentation", {})
    config["augmentation"]["enabled"] = True
    config["augmentation"]["flip_axis_prob"] = item["flip_axis_prob"]
    config["augmentation"]["rotate90_prob"] = item["rotate90_prob"]
    config["augmentation"]["standardize"] = True
    config["augmentation"]["standardize_channel_wise"] = True
    config["use_monai_transforms"] = False

    config_path = config_dir / f"{item['name']}.yml"
    with config_path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    rows.append(
        {
            "run": item["name"],
            "config": str(config_path),
            "model": item["model"],
            "base_features": item["base_features"],
            "feature_set": item["feature_set"],
            "apbs_variant": "apbs_full_signed",
            "description": item["description"],
        }
    )

plan_path = config_dir / "work8b_plan.tsv"
with plan_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["run", "config", "model", "base_features", "feature_set", "apbs_variant", "description"],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote Work8B plan: {plan_path}")
for row in rows:
    print(f"{row['run']}\t{row['model']}\tbase_features={row['base_features']}\t{row['description']}")
PY

PLAN_FILE="$CONFIG_DIR/work8b_plan.tsv"

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "Dry run enabled. Generated configs only; no training was started."
  exit 0
fi

tail -n +2 "$PLAN_FILE" | while IFS=$'\t' read -r run_name config_path model_name base_features feature_set apbs_variant description; do
  run_dir="$OUTPUT_ROOT/$run_name"
  log_path="$run_dir/log/training.log"
  final_model_path="$run_dir/weights/${model_name}_final_model.pth"

  echo
  echo "============================================================"
  echo "Running $run_name"
  echo "============================================================"
  echo "Model: $model_name | base_features: $base_features"
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
    --model "$model_name" \
    --base_features "$base_features" \
    --num_workers "$NUM_WORKERS" \
    --base_model_output_dir "$OUTPUT_ROOT"
done

"$PYTHON_BIN" scripts/summarize_gridfix_runs.py \
  --output-root "$OUTPUT_ROOT" \
  --write-csv "$OUTPUT_ROOT/run_summary.csv" | tee "$OUTPUT_ROOT/summary_stdout.txt"

echo
echo "Done."
echo "Summary CSV: $OUTPUT_ROOT/run_summary.csv"
