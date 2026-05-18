#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGURABLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/nfs/production/arl/chembl/tevfik/DEEP_APBS_DATASETS}"
H5_DIR="${H5_DIR:-$DATA_ROOT/cache/work11_cache_gridfix_v1/scpdb/label_cavity6/box36_span70}"
LABEL="${LABEL:-binding_site_cavity6}"
MANIFEST="${MANIFEST:-$H5_DIR/manifest.csv}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /homes/tevfik/PHD/generate_cache/.conda/bin/python ]]; then
    PYTHON_BIN="/homes/tevfik/PHD/generate_cache/.conda/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

if [[ -z "${TRAIN_PYTHON_BIN:-}" ]]; then
  if [[ -x "$CONFIGURABLE_DIR/../.venv/bin/python" ]]; then
    TRAIN_PYTHON_BIN="$CONFIGURABLE_DIR/../.venv/bin/python"
  elif [[ -x /homes/tevfik/PHD/3dunet-apbs/.venv/bin/python ]]; then
    TRAIN_PYTHON_BIN="/homes/tevfik/PHD/3dunet-apbs/.venv/bin/python"
  else
    TRAIN_PYTHON_BIN="$PYTHON_BIN"
  fi
fi

if [[ -z "${KALASANTY_FOLD_DIR:-}" ]]; then
  for candidate in \
    "/homes/tevfik/PHD/generate_cache/data/kalasanty" \
    "/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/generate_cache/data/kalasanty"; do
    if [[ -d "$candidate" ]]; then
      KALASANTY_FOLD_DIR="$candidate"
      break
    fi
  done
fi

if [[ -z "${PURESNET_ID_LIST:-}" ]]; then
  for candidate in \
    "/homes/tevfik/PHD/puresnet_scpdb_5020_ids.txt" \
    "/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/external/puresnet/puresnet_scpdb_5020_ids.txt"; do
    if [[ -f "$candidate" ]]; then
      PURESNET_ID_LIST="$candidate"
      break
    fi
  done
fi

BENCHMARKS="${BENCHMARKS:-kalasanty,puresnet}"
KALASANTY_FOLDS="${KALASANTY_FOLDS:-0,1,2,3,4,5,6,7,8,9}"
PURESNET_FOLDS="${PURESNET_FOLDS:-0,1,2,3}"
PURESNET_FOLD_COUNT="${PURESNET_FOLD_COUNT:-4}"
PURESNET_SPLIT_SEED="${PURESNET_SPLIT_SEED:-42}"

EPOCHS="${EPOCHS:-150}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-25}"
VALIDATION_THRESHOLD="${VALIDATION_THRESHOLD:-0.40}"
PAPER_METRICS_START_EPOCH="${PAPER_METRICS_START_EPOCH:-31}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-$BATCH_SIZE}"
MODEL="${MODEL:-ResNet3D4L}"
BASE_FEATURES="${BASE_FEATURES:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
LOSS_ALPHA="${LOSS_ALPHA:-0.5}"
POS_WEIGHT="${POS_WEIGHT:-2.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
HP_SUFFIX="${HP_SUFFIX:-lr1e4_alpha05_pos2_wd1e5}"

BASE_CONFIG="${BASE_CONFIG:-$CONFIGURABLE_DIR/config/local/scpdb_kalasanty36/scpdb_kalasanty36_apbs_compact_chem_fold0.yml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/runs/work14_scpdb_box36_span70_literature_benchmark_${MODEL}_apbs_bestfeatures_${EPOCHS}epoch_thr${VALIDATION_THRESHOLD/./}}"
CONFIG_DIR="${CONFIG_DIR:-$OUTPUT_ROOT/generated_configs}"
SPLIT_ROOT="${SPLIT_ROOT:-$H5_DIR/splits_literature_benchmark}"
KALASANTY_SPLIT_DIR="${KALASANTY_SPLIT_DIR:-$SPLIT_ROOT/kalasanty10_available_h5}"
PURESNET_SPLIT_DIR="${PURESNET_SPLIT_DIR:-$SPLIT_ROOT/puresnet5020_available_h5_kfold${PURESNET_FOLD_COUNT}_seed${PURESNET_SPLIT_SEED}}"
FEATURE_SET_FILTER="${FEATURE_SET_FILTER:-}"

SUBMIT="${SUBMIT:-0}"
SBATCH_ARRAY_LIMIT="${SBATCH_ARRAY_LIMIT:-4}"
JOB_NAME="${JOB_NAME:-scpdb36-lit}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY_GB="${MEMORY_GB:-80}"
WALLTIME="${WALLTIME:-3-00:00:00}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CLEAN_INCOMPLETE="${CLEAN_INCOMPLETE:-1}"

cd "$CONFIGURABLE_DIR"
mkdir -p "$OUTPUT_ROOT" "$CONFIG_DIR" "$SPLIT_ROOT"

required_features=(
  "electrostatic_grid_v1_ligand_proximal_chains_7A_full_signed150"
  "shape"
  "atomic_donor"
  "atomic_acceptor"
  "atomic_hydrophobic"
  "atomic_aromatic"
  "dist_to_surface"
)

contains_benchmark() {
  local needle="$1"
  IFS=',' read -r -a benchmark_items <<< "$BENCHMARKS"
  for item in "${benchmark_items[@]}"; do
    item="$(echo "$item" | xargs)"
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

echo "scPDB box36 literature benchmark config preparation"
echo "Repo: $CONFIGURABLE_DIR"
echo "H5 dir: $H5_DIR"
echo "Manifest: $MANIFEST"
echo "Label: $LABEL"
echo "Benchmarks: $BENCHMARKS"
echo "Kalasanty fold dir: ${KALASANTY_FOLD_DIR:-<missing>}"
echo "PUResNet ID list: ${PURESNET_ID_LIST:-<missing>}"
echo "Output root: $OUTPUT_ROOT"
echo "Generated configs: $CONFIG_DIR"
echo "Model: $MODEL"
echo "Base features: $BASE_FEATURES"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Hyperparameters: lr=$LEARNING_RATE alpha=$LOSS_ALPHA pos_weight=$POS_WEIGHT wd=$WEIGHT_DECAY"
echo "Config/split Python: $PYTHON_BIN"
echo "Training Python: $TRAIN_PYTHON_BIN"
echo "Submit: $SUBMIT"

common_split_args=(
  --h5-dir "$H5_DIR"
  --label "$LABEL"
  --allow-label-atoms-outside-box
)
if [[ -f "$MANIFEST" ]]; then
  common_split_args+=(--manifest "$MANIFEST")
fi
for feature_name in "${required_features[@]}"; do
  common_split_args+=(--required-feature "$feature_name")
done

if contains_benchmark "kalasanty"; then
  if [[ -z "${KALASANTY_FOLD_DIR:-}" || ! -d "$KALASANTY_FOLD_DIR" ]]; then
    echo "Kalasanty fold dir not found. Set KALASANTY_FOLD_DIR."
    exit 1
  fi
  echo
  echo "Building Kalasanty 10-fold splits from official fold ID files"
  "$PYTHON_BIN" scripts/build_cache_splits_from_fold_ids.py \
    "${common_split_args[@]}" \
    --h5-exists-only \
    --fold-dir "$KALASANTY_FOLD_DIR" \
    --output-dir "$KALASANTY_SPLIT_DIR" \
    --folds "$KALASANTY_FOLDS" \
    --train-pattern "train_ids_fold{fold}" \
    --validation-pattern "test_ids_fold{fold}" \
    --allow-missing-fold-ids
fi

if contains_benchmark "puresnet"; then
  if [[ -z "${PURESNET_ID_LIST:-}" || ! -f "$PURESNET_ID_LIST" ]]; then
    echo "PUResNet ID list not found. Set PURESNET_ID_LIST."
    exit 1
  fi
  echo
  echo "Building PUResNet 5020-derived deterministic 4-fold splits from available H5 cases"
  "$PYTHON_BIN" scripts/build_cache_kfold_splits_from_id_list.py \
    "${common_split_args[@]}" \
    --h5-exists-only \
    --id-list "$PURESNET_ID_LIST" \
    --output-dir "$PURESNET_SPLIT_DIR" \
    --folds "$PURESNET_FOLD_COUNT" \
    --seed "$PURESNET_SPLIT_SEED"
fi

export BASE_CONFIG CONFIG_DIR OUTPUT_ROOT H5_DIR LABEL
export EPOCHS EARLY_STOPPING_PATIENCE VALIDATION_THRESHOLD PAPER_METRICS_START_EPOCH
export BATCH_SIZE VALIDATION_BATCH_SIZE LEARNING_RATE LOSS_ALPHA POS_WEIGHT WEIGHT_DECAY HP_SUFFIX
export BENCHMARKS KALASANTY_FOLDS PURESNET_FOLDS KALASANTY_SPLIT_DIR PURESNET_SPLIT_DIR FEATURE_SET_FILTER
"$PYTHON_BIN" - <<'PY'
import copy
import csv
import os
from collections import OrderedDict
from pathlib import Path

import yaml


def parse_items(value):
    return [item.strip() for item in value.split(",") if item.strip()]


base_config_path = Path(os.environ["BASE_CONFIG"])
config_dir = Path(os.environ["CONFIG_DIR"])
output_root = Path(os.environ["OUTPUT_ROOT"])
h5_dir = os.environ["H5_DIR"]
label = os.environ["LABEL"]
epochs = int(os.environ["EPOCHS"])
early_stopping_patience = int(os.environ["EARLY_STOPPING_PATIENCE"])
validation_threshold = float(os.environ["VALIDATION_THRESHOLD"])
paper_metrics_start_epoch = int(os.environ["PAPER_METRICS_START_EPOCH"])
batch_size = int(os.environ["BATCH_SIZE"])
validation_batch_size = int(os.environ["VALIDATION_BATCH_SIZE"])
learning_rate = float(os.environ["LEARNING_RATE"])
loss_alpha = float(os.environ["LOSS_ALPHA"])
pos_weight = float(os.environ["POS_WEIGHT"])
weight_decay = float(os.environ["WEIGHT_DECAY"])
hp_suffix = os.environ["HP_SUFFIX"]
benchmarks = parse_items(os.environ["BENCHMARKS"])
feature_set_filter = set(parse_items(os.environ.get("FEATURE_SET_FILTER", "")))

feature_sets = OrderedDict(
    [
        (
            "apbs_v1_full_signed_shape",
            [
                "electrostatic_grid_v1_ligand_proximal_chains_7A_full_signed150",
                "shape",
            ],
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

if feature_set_filter:
    feature_sets = OrderedDict(
        (name, features) for name, features in feature_sets.items() if name in feature_set_filter
    )
if not feature_sets:
    raise SystemExit("No feature sets selected.")

with base_config_path.open() as handle:
    base_config = yaml.safe_load(handle)

benchmark_defs = []
if "kalasanty" in benchmarks:
    benchmark_defs.append(
        {
            "name": "kalasanty10",
            "split_dir": Path(os.environ["KALASANTY_SPLIT_DIR"]),
            "folds": [int(item) for item in parse_items(os.environ["KALASANTY_FOLDS"])],
            "split_policy": "official Kalasanty scPDB 10-fold IDs; unavailable H5 cases excluded",
        }
    )
if "puresnet" in benchmarks:
    benchmark_defs.append(
        {
            "name": "puresnet5020_kfold4",
            "split_dir": Path(os.environ["PURESNET_SPLIT_DIR"]),
            "folds": [int(item) for item in parse_items(os.environ["PURESNET_FOLDS"])],
            "split_policy": "PUResNet 5020 ID list filtered to available H5 cases; deterministic 4-fold, not exact paper folds",
        }
    )
if not benchmark_defs:
    raise SystemExit("No supported benchmark selected.")

config_dir.mkdir(parents=True, exist_ok=True)
threshold_sweep = sorted(
    {0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, validation_threshold}
)

requested_runs = [
    (benchmark, fold_idx, feature_name, features)
    for benchmark in benchmark_defs
    for fold_idx in benchmark["folds"]
    for feature_name, features in feature_sets.items()
]
total_run_count = len(requested_runs)
config_paths = []
metadata_rows = []

for run_index, (benchmark, fold_idx, feature_name, features) in enumerate(requested_runs, start=1):
    split_dir = benchmark["split_dir"]
    train_file = split_dir / f"fold{fold_idx}_train_cases.txt"
    validation_file = split_dir / f"fold{fold_idx}_validation_cases.txt"
    if not train_file.exists() or not validation_file.exists():
        raise SystemExit(f"Missing split files for {benchmark['name']} fold {fold_idx} under {split_dir}")

    config = copy.deepcopy(base_config)
    run_name = f"scpdb_box36_span70_{benchmark['name']}_fold{fold_idx}_{feature_name}_{hp_suffix}"
    config["name"] = run_name
    config["h5_directory"] = h5_dir
    config["label"] = label
    config["features"] = features
    config["feature_set"] = {
        "name": f"{benchmark['name']}_fold{fold_idx}_{feature_name}",
        "benchmark": benchmark["name"],
        "feature_name": feature_name,
        "fold": fold_idx,
        "index": run_index,
        "count": total_run_count,
        "split_policy": benchmark["split_policy"],
    }
    config["training"]["num_epochs"] = epochs
    config["training"]["early_stopping_patience"] = early_stopping_patience
    config["training"]["batch_size"] = batch_size
    config["training"]["learning_rate"] = learning_rate
    config["training"]["weight_decay"] = weight_decay
    config["training"].setdefault("loss", {})
    config["training"]["loss"]["alpha"] = loss_alpha
    config["training"]["loss"]["pos_weight"] = pos_weight
    config["training"]["loss"]["dynamic_pos_weight"] = False
    config["validation"]["batch_size"] = validation_batch_size
    config["validation"]["threshold"] = validation_threshold
    config["validation"]["threshold_sweep"] = threshold_sweep
    config["validation"].setdefault("paper_metrics", {})
    paper_metrics = config["validation"]["paper_metrics"]
    paper_metrics["enabled"] = True
    paper_metrics["dcc_reference"] = "label_center"
    paper_metrics["dcc_cutoff_angstrom"] = 4.0
    paper_metrics["dca_cutoff_angstrom"] = 4.0
    paper_metrics["min_component_voxels"] = 5
    paper_metrics["min_component_volume_angstrom3"] = 50.0
    paper_metrics["postprocess"] = "raw"
    paper_metrics["comparison_postprocess"] = ["kalasanty_puresnet"]
    paper_metrics["selection_metric"] = "dcc_voxel_dca_dvo_volume"
    paper_metrics["selection_dvo_weight"] = 1.0
    paper_metrics["selection_voxel_f1_weight"] = 1.0
    paper_metrics["selection_dca_weight"] = 0.25
    paper_metrics["selection_no_dcc_score_scale"] = 0.05
    paper_metrics["selection_max_mean_predicted_positive_voxels"] = 5000
    paper_metrics["full_evaluation_start_epoch"] = paper_metrics_start_epoch
    paper_metrics["top_k"] = [1, 3]
    config.setdefault("datasets", {})
    config["datasets"]["train_file"] = str(train_file)
    config["datasets"]["validation_file"] = str(validation_file)

    config_path = config_dir / f"{run_name}.yml"
    with config_path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    config_paths.append(config_path)
    metadata_rows.append(
        {
            "run": run_name,
            "benchmark": benchmark["name"],
            "fold": fold_idx,
            "feature_set_name": feature_name,
            "hyperparameter_name": hp_suffix,
            "run_index": run_index,
            "run_count": total_run_count,
            "feature_count": len(features),
            "features": ",".join(features),
            "train_file": str(train_file),
            "validation_file": str(validation_file),
            "split_policy": benchmark["split_policy"],
            "learning_rate": learning_rate,
            "loss_alpha": loss_alpha,
            "pos_weight": pos_weight,
            "weight_decay": weight_decay,
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

metadata_path = config_dir / "benchmark_runs.csv"
with metadata_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0]))
    writer.writeheader()
    writer.writerows(metadata_rows)

print(f"Wrote {len(config_paths)} configs")
print(f"Config list: {list_path}")
print(f"Run metadata: {metadata_path}")
print(f"Output root: {output_root}")
PY

CONFIG_LIST="$CONFIG_DIR/config_list.txt"
CONFIG_COUNT="$(wc -l < "$CONFIG_LIST" | xargs)"
echo
echo "Config count: $CONFIG_COUNT"
echo "Config list: $CONFIG_LIST"

if [[ "$SUBMIT" != "1" ]]; then
  echo "Submission skipped. Set SUBMIT=1 to submit the Slurm array."
  exit 0
fi

echo "Checking training Python can import torch"
"$TRAIN_PYTHON_BIN" - <<'PY'
import torch
print(f"torch {torch.__version__}")
PY

mkdir -p "$OUTPUT_ROOT/slurm"
echo "Submitting Slurm array 1-${CONFIG_COUNT}%${SBATCH_ARRAY_LIMIT}"
sbatch \
  --job-name "$JOB_NAME" \
  --array "1-${CONFIG_COUNT}%${SBATCH_ARRAY_LIMIT}" \
  --gres "gpu:${GPU_TYPE}:1" \
  --cpus-per-task "$CPUS_PER_TASK" \
  --mem "${MEMORY_GB}G" \
  --time "$WALLTIME" \
  --output "$OUTPUT_ROOT/slurm/%x_%A_%a.out" \
  --error "$OUTPUT_ROOT/slurm/%x_%A_%a.err" \
  --export "ALL,CONFIG_LIST=$CONFIG_LIST,OUTPUT_ROOT=$OUTPUT_ROOT,REPO_DIR=$CONFIGURABLE_DIR,PYTHON_BIN=$TRAIN_PYTHON_BIN,MODEL=$MODEL,BASE_FEATURES=$BASE_FEATURES,NUM_WORKERS=$NUM_WORKERS,SKIP_COMPLETED=$SKIP_COMPLETED,CLEAN_INCOMPLETE=$CLEAN_INCOMPLETE" \
  "$CONFIGURABLE_DIR/slurm/run_config_array.sh"
