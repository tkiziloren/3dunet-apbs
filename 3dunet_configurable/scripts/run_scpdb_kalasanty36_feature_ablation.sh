#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGURABLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

EPOCHS="${EPOCHS:-100}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
VALIDATION_THRESHOLD="${VALIDATION_THRESHOLD:-0.40}"
MODEL="${MODEL:-UNet3D4LStrided}"
BASE_FEATURES="${BASE_FEATURES:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PYTHON_BIN="${PYTHON_BIN:-$CONFIGURABLE_DIR/../.venv/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-$CONFIGURABLE_DIR/config/local/scpdb_kalasanty36/scpdb_kalasanty36_apbs_compact_chem_fold0.yml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/Users/tevfik/Sandbox/github/PHD/runs/scpdb_kalasanty36_feature_ablation_${EPOCHS}epoch_thr${VALIDATION_THRESHOLD/./}}"
CONFIG_DIR="${CONFIG_DIR:-$OUTPUT_ROOT/generated_configs}"
RUN_FILTER="${RUN_FILTER:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CLEAN_INCOMPLETE="${CLEAN_INCOMPLETE:-1}"
DRY_RUN="${DRY_RUN:-0}"

cd "$CONFIGURABLE_DIR"
mkdir -p "$OUTPUT_ROOT" "$CONFIG_DIR"

echo "Feature ablation output root: $OUTPUT_ROOT"
echo "Generated configs: $CONFIG_DIR"
echo "Python: $PYTHON_BIN"
echo "Model: $MODEL"
echo "Base features: $BASE_FEATURES"
echo "Epochs: $EPOCHS"
echo "Early stopping patience: $EARLY_STOPPING_PATIENCE"
echo "Fixed validation threshold: $VALIDATION_THRESHOLD"
echo "Run filter: ${RUN_FILTER:-<all>}"
echo "Clean incomplete runs: $CLEAN_INCOMPLETE"
echo "Dry run: $DRY_RUN"

export BASE_CONFIG CONFIG_DIR EPOCHS EARLY_STOPPING_PATIENCE VALIDATION_THRESHOLD RUN_FILTER
"$PYTHON_BIN" - <<'PY'
import copy
import csv
import os
from collections import OrderedDict
from pathlib import Path

import yaml

base_config_path = Path(os.environ["BASE_CONFIG"])
config_dir = Path(os.environ["CONFIG_DIR"])
epochs = int(os.environ["EPOCHS"])
early_stopping_patience = int(os.environ["EARLY_STOPPING_PATIENCE"])
validation_threshold = float(os.environ["VALIDATION_THRESHOLD"])
run_filter = {
    item.strip()
    for item in os.environ.get("RUN_FILTER", "").split(",")
    if item.strip()
}

feature_sets = OrderedDict(
    [
        ("shape_only", ["shape"]),
        ("apbs_only", ["electrostatic_grid"]),
        ("apbs_shape", ["electrostatic_grid", "shape"]),
        (
            "shape_selected_chem",
            ["shape", "atomic_donor", "atomic_acceptor", "atomic_hydrophobic", "atomic_aromatic"],
        ),
        (
            "apbs_shape_selected_chem",
            [
                "electrostatic_grid",
                "shape",
                "atomic_donor",
                "atomic_acceptor",
                "atomic_hydrophobic",
                "atomic_aromatic",
            ],
        ),
        (
            "apbs_shape_selected_chem_surface_hydro",
            [
                "electrostatic_grid",
                "shape",
                "atomic_donor",
                "atomic_acceptor",
                "atomic_hydrophobic",
                "atomic_aromatic",
                "hydrophobicity",
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
threshold_sweep = [0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]
threshold_sweep = sorted({float(value) for value in threshold_sweep + [validation_threshold]})

feature_set_count = len(feature_sets)

for feature_set_index, (suffix, features) in enumerate(feature_sets.items(), start=1):
    if run_filter and suffix not in run_filter:
        continue
    leaked = forbidden_features.intersection(features)
    if leaked:
        raise SystemExit(f"Feature set {suffix} contains forbidden leakage features: {sorted(leaked)}")

    config = copy.deepcopy(base_config)
    config["name"] = f"scpdb_kalasanty36_ablation_{suffix}"
    config["feature_set"] = {
        "name": suffix,
        "index": feature_set_index,
        "count": feature_set_count,
    }
    config["features"] = features
    config["training"]["num_epochs"] = epochs
    config["training"]["early_stopping_patience"] = early_stopping_patience
    config["validation"]["threshold"] = validation_threshold
    config["validation"]["threshold_sweep"] = threshold_sweep

    config_path = config_dir / f"{config['name']}.yml"
    with config_path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    config_paths.append(config_path)
    metadata_rows.append(
        {
            "run": config_path.stem,
            "feature_set_name": suffix,
            "feature_set_index": feature_set_index,
            "feature_set_count": feature_set_count,
            "features": ",".join(features),
            "feature_count": len(features),
            "fixed_validation_threshold": validation_threshold,
            "epochs": epochs,
            "early_stopping_patience": early_stopping_patience,
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
            "feature_set_name",
            "feature_set_index",
            "feature_set_count",
            "feature_count",
            "features",
            "fixed_validation_threshold",
            "epochs",
            "early_stopping_patience",
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
echo "============================================================"
echo "Summarizing feature ablation runs"
echo "============================================================"
"$PYTHON_BIN" scripts/summarize_gridfix_runs.py \
  --output-root "$OUTPUT_ROOT" \
  --write-csv "$OUTPUT_ROOT/run_summary.csv" | tee "$OUTPUT_ROOT/summary_stdout.txt"

echo
echo "Done."
echo "Summary CSV: $OUTPUT_ROOT/run_summary.csv"
echo "Master output can be redirected to: $OUTPUT_ROOT/master.log"
