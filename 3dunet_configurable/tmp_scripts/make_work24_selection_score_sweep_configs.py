#!/usr/bin/env python3
"""Generate PUResNet5020 fold0 selection-score sweep configs.

This produces training configs only. It does not submit Slurm jobs.
"""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
BASE_CONFIG_CANDIDATES = [
    REPO
    / "reports/work14_and_pdbbind_reeval_2026-05-19/remote_csv/scpdb_puresnet/"
    / "scpdb_box36_span70_puresnet5020_kfold4_fold0_apbs_v1_full_signed_shape_selected_chem_surface_lr1e4_alpha05_pos2_wd1e5"
    / "config_snapshot.yml",
    Path(
        "/nfs/production/arl/chembl/tevfik/DEEP_APBS_DATASETS/runs/"
        "work20_scpdb_box36_span70_puresnet5020_kfold4_unetplusplus_surface_250epoch_thr040/"
        "scpdb_box36_span70_puresnet5020_kfold4_fold0_apbs_v1_full_signed_shape_selected_chem_surface_lr1e4_alpha05_pos2_wd1e5/"
        "config_snapshot.yml"
    ),
]
WORK_NAME = "work24_puresnet5020_fold0_unetplusplus_selection_score_sweep_250epoch_thr040"
OUT_DIR = REPO / "reports" / WORK_NAME
CONFIG_DIR = OUT_DIR / "generated_configs"


def profile_rows():
    rows = []

    def add(family, f1, dcc, dca, dvo, pli, cap, no_pred, f1_floor, volume_power=1.0):
        index = len(rows) + 1
        rows.append(
            {
                "index": index,
                "profile": f"{family}_{index:02d}",
                "family": family,
                "weights": {
                    "paper_f1": f1,
                    "dcc_success_rate_4a": dcc,
                    "dca_success_rate_4a": dca,
                    "mean_dvo_dcc_success": dvo,
                    "mean_pli_dcc_success": pli,
                    "predicted_rate": 0.10,
                },
                "cap": cap,
                "no_prediction_weight": no_pred,
                "f1_floor": f1_floor,
                "floor_scale": 0.20,
                "volume_power": volume_power,
            }
        )

    for dvo in (0.75, 1.00, 1.25):
        for pli in (0.25, 0.50):
            for cap in (5000, 8000):
                add("balanced_dvo_pli", 1.50, 1.00, 0.35, dvo, pli, cap, 0.20, 0.72)

    for dvo in (1.50, 2.00, 2.50):
        for pli in (0.25, 0.50):
            for cap in (5000, 8000):
                add("dvo_priority", 1.75, 1.00, 0.25, dvo, pli, cap, 0.15, 0.72)

    for dvo in (0.75, 1.00):
        for pli in (0.75, 1.25, 1.75):
            for cap in (5000, 8000):
                add("pli_priority", 1.75, 1.00, 0.25, dvo, pli, cap, 0.15, 0.72)

    for dcc in (1.00, 1.25):
        for dca in (0.35, 0.50):
            for dvo in (1.00, 1.50, 2.00):
                add("compact_localization", 2.00, dcc, dca, dvo, 0.50, 3500, 0.25, 0.73, 1.25)

    assert len(rows) == 48
    return rows


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    base_config = next((path for path in BASE_CONFIG_CANDIDATES if path.exists()), None)
    if base_config is None:
        checked = "\n".join(str(path) for path in BASE_CONFIG_CANDIDATES)
        raise FileNotFoundError(f"No base config found. Checked:\n{checked}")
    with base_config.open() as handle:
        base = yaml.safe_load(handle)

    config_paths = []
    manifest_rows = []
    rows = profile_rows()
    for row in rows:
        cfg = deepcopy(base)
        suffix = row["profile"].replace(".", "p")
        run_name = (
            "scpdb_box36_span70_puresnet5020_fold0_unetplusplus_"
            f"apbs_shape_selected_chem_surface_sel_{suffix}"
        )
        cfg["name"] = run_name
        cfg.setdefault("training", {})["num_epochs"] = 250
        cfg["training"]["early_stopping_patience"] = 35
        cfg.setdefault("feature_set", {})
        cfg["feature_set"].update(
            {
                "name": "puresnet5020_fold0_unetplusplus_apbs_shape_selected_chem_surface_selection_score_sweep",
                "benchmark": "puresnet5020_kfold4",
                "feature_name": "apbs_v1_full_signed_shape_selected_chem_surface",
                "fold": 0,
                "index": row["index"],
                "count": len(rows),
                "selection_profile": row["profile"],
                "model": "UNetPlusPlus3D",
            }
        )

        paper_metrics = cfg.setdefault("validation", {}).setdefault("paper_metrics", {})
        paper_metrics.update(
            {
                "selection_profile": row["profile"],
                "selection_metric": "weighted_sum",
                "selection_weights": row["weights"],
                "selection_no_prediction_weight": row["no_prediction_weight"],
                "selection_max_mean_predicted_positive_voxels": row["cap"],
                "selection_volume_penalty_power": row["volume_power"],
                "selection_min_paper_f1": row["f1_floor"],
                "selection_below_min_paper_f1_score_scale": row["floor_scale"],
            }
        )

        out_path = CONFIG_DIR / f"{run_name}.yml"
        with out_path.open("w") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        config_paths.append(out_path)

        manifest_rows.append(
            {
                "index": row["index"],
                "profile": row["profile"],
                "family": row["family"],
                "config": str(out_path),
                "paper_f1_weight": row["weights"]["paper_f1"],
                "dcc_weight": row["weights"]["dcc_success_rate_4a"],
                "dca_weight": row["weights"]["dca_success_rate_4a"],
                "dvo_weight": row["weights"]["mean_dvo_dcc_success"],
                "pli_weight": row["weights"]["mean_pli_dcc_success"],
                "predicted_rate_weight": row["weights"]["predicted_rate"],
                "no_prediction_weight": row["no_prediction_weight"],
                "volume_cap": row["cap"],
                "volume_penalty_power": row["volume_power"],
                "min_paper_f1": row["f1_floor"],
                "below_min_paper_f1_score_scale": row["floor_scale"],
            }
        )

    config_list = OUT_DIR / "config_list.txt"
    config_list.write_text("\n".join(str(path) for path in config_paths) + "\n")

    manifest = OUT_DIR / "selection_score_profiles.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(config_paths)} configs")
    print(f"Config list: {config_list}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
