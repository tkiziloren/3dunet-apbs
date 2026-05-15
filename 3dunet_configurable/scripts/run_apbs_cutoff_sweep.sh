#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGURABLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

EPOCHS="${EPOCHS:-150}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
VALIDATION_THRESHOLD="${VALIDATION_THRESHOLD:-0.40}"
MODEL="${MODEL:-UNet3D4LStrided}"
BASE_FEATURES="${BASE_FEATURES:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PYTHON_BIN="${PYTHON_BIN:-$CONFIGURABLE_DIR/../.venv/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-$CONFIGURABLE_DIR/config/local/scpdb_kalasanty36/scpdb_kalasanty36_apbs_compact_chem_fold0.yml}"
SPLIT_DIR="${SPLIT_DIR:-/Users/tevfik/Sandbox/github/PHD/data/scPDB_cache_gridfix_v1/label_cavity6/box36_span70/splits_cache_kfold5_seed42}"
FOLDS="${FOLDS:-0}"
FEATURE_SET="${FEATURE_SET:-apbs_only}"
CUTOFF_VARIANTS="${CUTOFF_VARIANTS:-}"
STANDARDIZE_CHANNEL_WISE="${STANDARDIZE_CHANNEL_WISE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/Users/tevfik/Sandbox/github/PHD/runs/apbs_cutoff_sweep_${FEATURE_SET}_fold${FOLDS//,/}_150epoch_thr${VALIDATION_THRESHOLD/./}}"
CONFIG_DIR="${CONFIG_DIR:-$OUTPUT_ROOT/generated_configs}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CLEAN_INCOMPLETE="${CLEAN_INCOMPLETE:-1}"
DRY_RUN="${DRY_RUN:-0}"

cd "$CONFIGURABLE_DIR"
mkdir -p "$OUTPUT_ROOT" "$CONFIG_DIR"

echo "APBS cutoff sweep output root: $OUTPUT_ROOT"
echo "Feature set: $FEATURE_SET"
echo "Generated configs: $CONFIG_DIR"
echo "Split dir: $SPLIT_DIR"
echo "Folds: $FOLDS"
echo "Cutoff variants: ${CUTOFF_VARIANTS:-<all>}"
echo "Python: $PYTHON_BIN"
echo "Model: $MODEL"
echo "Base features: $BASE_FEATURES"
echo "Epochs: $EPOCHS"
echo "Early stopping patience: $EARLY_STOPPING_PATIENCE"
echo "Fixed validation threshold: $VALIDATION_THRESHOLD"
echo "Standardize channel-wise override: ${STANDARDIZE_CHANNEL_WISE:-<from base config>}"
echo "Dry run: $DRY_RUN"

export BASE_CONFIG CONFIG_DIR EPOCHS EARLY_STOPPING_PATIENCE VALIDATION_THRESHOLD SPLIT_DIR FOLDS FEATURE_SET CUTOFF_VARIANTS STANDARDIZE_CHANNEL_WISE
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
epochs = int(os.environ["EPOCHS"])
early_stopping_patience = int(os.environ["EARLY_STOPPING_PATIENCE"])
validation_threshold = float(os.environ["VALIDATION_THRESHOLD"])
folds = [int(item.strip()) for item in os.environ["FOLDS"].split(",") if item.strip()]
feature_set_name = os.environ["FEATURE_SET"]
cutoff_variant_filter = {
    item.strip()
    for item in os.environ.get("CUTOFF_VARIANTS", "").split(",")
    if item.strip()
}
standardize_channel_wise_override = os.environ.get("STANDARDIZE_CHANNEL_WISE", "").strip()

feature_sets = {
    "apbs_only": ["electrostatic_grid"],
    "apbs_shape": ["electrostatic_grid", "shape"],
    "apbs_selected_chem": [
        "electrostatic_grid",
        "atomic_donor",
        "atomic_acceptor",
        "atomic_hydrophobic",
        "atomic_aromatic",
    ],
    "apbs_shape_selected_chem": [
        "electrostatic_grid",
        "shape",
        "atomic_donor",
        "atomic_acceptor",
        "atomic_hydrophobic",
        "atomic_aromatic",
    ],
}
if feature_set_name not in feature_sets:
    raise SystemExit(f"Unsupported FEATURE_SET={feature_set_name}. Choose one of: {', '.join(sorted(feature_sets))}")

cutoff_variants = OrderedDict(
    [
        (
            "apbs_clip5_minmax",
            {
                "description": "APBS clipped to [-5, 5], then normalized to [0, 1]",
                "normalization": {
                    "electrostatic_grid": {"min": -5.0, "max": 5.0, "clip": True, "normalize": True}
                },
            },
        ),
        (
            "apbs_clip10_minmax",
            {
                "description": "APBS clipped to [-10, 10], then normalized to [0, 1]",
                "normalization": {
                    "electrostatic_grid": {"min": -10.0, "max": 10.0, "clip": True, "normalize": True}
                },
            },
        ),
        (
            "apbs_clip10",
            {
                "description": "Alias for apbs_clip10_minmax",
                "normalization": {
                    "electrostatic_grid": {"min": -10.0, "max": 10.0, "clip": True, "normalize": True}
                },
            },
        ),
        (
            "apbs_clip20_minmax",
            {
                "description": "APBS clipped to [-20, 20], then normalized to [0, 1]",
                "normalization": {
                    "electrostatic_grid": {"min": -20.0, "max": 20.0, "clip": True, "normalize": True}
                },
            },
        ),
        (
            "apbs_clip20",
            {
                "description": "Alias for apbs_clip20_minmax",
                "normalization": {
                    "electrostatic_grid": {"min": -20.0, "max": 20.0, "clip": True, "normalize": True}
                },
            },
        ),
        (
            "apbs_no_cutoff_current",
            {
                "description": "Raw APBS values, no clipping and no range normalization; training transform still standardizes input",
                "normalization": {
                    "electrostatic_grid": {"clip": False, "normalize": False}
                },
            },
        ),
        (
            "apbs_no_cutoff",
            {
                "description": "Alias for apbs_no_cutoff_current",
                "normalization": {
                    "electrostatic_grid": {"clip": False, "normalize": False}
                },
            },
        ),
        (
            "apbs_full_minmax",
            {
                "description": "APBS linearly scaled from [-150, 150] to [0, 1] without clipping",
                "normalization": {
                    "electrostatic_grid": {"min": -150.0, "max": 150.0, "clip": False, "normalize": True}
                },
            },
        ),
        (
            "apbs_full_signed",
            {
                "description": "APBS linearly scaled from [-150, 150] to [-1, 1] without clipping",
                "normalization": {
                    "electrostatic_grid": {
                        "min": -150.0,
                        "max": 150.0,
                        "clip": False,
                        "normalize": True,
                        "output_min": -1.0,
                        "output_max": 1.0,
                    }
                },
            },
        ),
        (
            "apbs_clip20_signed",
            {
                "description": "APBS clipped to [-20, 20], then normalized to [-1, 1]",
                "normalization": {
                    "electrostatic_grid": {
                        "min": -20.0,
                        "max": 20.0,
                        "clip": True,
                        "normalize": True,
                        "output_min": -1.0,
                        "output_max": 1.0,
                    }
                },
            },
        ),
        (
            "apbs_posneg_clip20",
            {
                "description": "APBS split into two channels: positive magnitude and negative magnitude, each clipped to [0, 20] and scaled to [0, 1]",
                "normalization": {},
                "apbs_feature_override": ["electrostatic_positive_clip20", "electrostatic_negative_clip20"],
            },
        ),
    ]
)
unknown_variants = cutoff_variant_filter.difference(cutoff_variants)
if unknown_variants:
    raise SystemExit(
        f"Unsupported CUTOFF_VARIANTS={sorted(unknown_variants)}. "
        f"Choose from: {', '.join(cutoff_variants)}"
    )
if cutoff_variant_filter:
    cutoff_variants = OrderedDict(
        (name, variant)
        for name, variant in cutoff_variants.items()
        if name in cutoff_variant_filter
    )

with base_config_path.open() as handle:
    base_config = yaml.safe_load(handle)

config_dir.mkdir(parents=True, exist_ok=True)
config_paths = []
metadata_rows = []
threshold_sweep = [0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]
threshold_sweep = sorted({float(value) for value in threshold_sweep + [validation_threshold]})

requested_runs = [
    (fold_idx, variant_name, variant)
    for fold_idx in folds
    for variant_name, variant in cutoff_variants.items()
]
total_run_count = len(requested_runs)

for global_run_index, (fold_idx, variant_name, variant) in enumerate(requested_runs, start=1):
    train_file = split_dir / f"fold{fold_idx}_train_cases.txt"
    validation_file = split_dir / f"fold{fold_idx}_validation_cases.txt"
    if not train_file.exists() or not validation_file.exists():
        raise SystemExit(f"Missing split files for fold {fold_idx} under {split_dir}")

    config = copy.deepcopy(base_config)
    config["name"] = f"scpdb_apbs_cutoff_fold{fold_idx}_{feature_set_name}_{variant_name}"
    config["feature_set"] = {
        "name": f"fold{fold_idx}_{feature_set_name}_{variant_name}",
        "feature_name": feature_set_name,
        "apbs_cutoff_variant": variant_name,
        "fold": fold_idx,
        "index": global_run_index,
        "count": total_run_count,
    }
    selected_features = list(feature_sets[feature_set_name])
    apbs_override = variant.get("apbs_feature_override")
    if apbs_override:
        selected_features = list(apbs_override) + [feat for feat in selected_features if feat != "electrostatic_grid"]
    config["features"] = selected_features
    config["feature_normalization"] = variant["normalization"]
    config.setdefault("metadata", {})
    config["metadata"]["apbs_cutoff_description"] = variant["description"]
    config["training"]["num_epochs"] = epochs
    config["training"]["early_stopping_patience"] = early_stopping_patience
    config["validation"]["threshold"] = validation_threshold
    config["validation"]["threshold_sweep"] = threshold_sweep
    config["datasets"]["train_file"] = str(train_file)
    config["datasets"]["validation_file"] = str(validation_file)
    if standardize_channel_wise_override:
        config.setdefault("augmentation", {})
        config["augmentation"]["standardize_channel_wise"] = standardize_channel_wise_override.lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

    config_path = config_dir / f"{config['name']}.yml"
    with config_path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    config_paths.append(config_path)
    metadata_rows.append(
        {
            "run": config_path.stem,
            "fold": fold_idx,
            "feature_set_name": feature_set_name,
            "apbs_cutoff_variant": variant_name,
            "description": variant["description"],
            "run_index": global_run_index,
            "run_count": total_run_count,
            "features": ",".join(config["features"]),
            "feature_normalization": variant["normalization"],
            "train_file": str(train_file),
            "validation_file": str(validation_file),
            "fixed_validation_threshold": validation_threshold,
            "epochs": epochs,
            "early_stopping_patience": early_stopping_patience,
            "standardize_channel_wise": config.get("augmentation", {}).get("standardize_channel_wise"),
        }
    )

list_path = config_dir / "config_list.txt"
list_path.write_text("\n".join(str(path) for path in config_paths) + "\n")

metadata_path = config_dir / "apbs_cutoff_variants.csv"
with metadata_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "run",
            "fold",
            "feature_set_name",
            "apbs_cutoff_variant",
            "description",
            "run_index",
            "run_count",
            "features",
            "feature_normalization",
            "train_file",
            "validation_file",
            "fixed_validation_threshold",
            "epochs",
            "early_stopping_patience",
            "standardize_channel_wise",
        ],
    )
    writer.writeheader()
    writer.writerows(metadata_rows)

print(f"Wrote {len(config_paths)} configs")
print(f"Config list: {list_path}")
print(f"Variant metadata: {metadata_path}")
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
echo "============================================================"
echo "Summarizing APBS cutoff sweep runs"
echo "============================================================"
"$PYTHON_BIN" scripts/summarize_gridfix_runs.py \
  --output-root "$OUTPUT_ROOT" \
  --write-csv "$OUTPUT_ROOT/run_summary.csv" | tee "$OUTPUT_ROOT/summary_stdout.txt"

echo
echo "Done."
echo "Summary CSV: $OUTPUT_ROOT/run_summary.csv"
echo "Master output can be redirected to: $OUTPUT_ROOT/master.log"
