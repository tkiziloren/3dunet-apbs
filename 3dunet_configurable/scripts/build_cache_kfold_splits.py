import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict

import h5py
import numpy as np


DEFAULT_H5_DIR = "/Users/tevfik/Sandbox/github/PHD/data/scPDB_cache_gridfix_v1/label_cavity6/box36_span70"
DEFAULT_LABEL = "binding_site_cavity6"
DEFAULT_REQUIRED_FEATURES = [
    "electrostatic_grid",
    "shape",
    "atomic_donor",
    "atomic_acceptor",
    "atomic_hydrophobic",
    "atomic_aromatic",
    "hydrophobicity",
    "dist_to_surface",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build balanced base-id grouped k-fold splits from the available H5 cache. "
            "All entries sharing the same scPDB/PDB base id stay in the same fold."
        )
    )
    parser.add_argument("--h5-dir", default=DEFAULT_H5_DIR)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--required-feature",
        action="append",
        dest="required_features",
        default=None,
        help="Feature that must exist in every included H5. Repeatable.",
    )
    parser.add_argument(
        "--allow-label-atoms-outside-box",
        action="store_true",
        help="Do not exclude manifest rows where label atoms are partly outside the grid box.",
    )
    parser.add_argument(
        "--allow-missing-ligand-mask",
        action="store_true",
        help="Do not require features/ligand for DCC/DCA reference metrics.",
    )
    return parser.parse_args()


def strip_scpdb_suffix(case_id):
    return re.sub(r"_[0-9]+$", "", case_id)


def read_manifest_rows(path):
    if path is None or not os.path.exists(path):
        return {}
    rows = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            case = row.get("case")
            if case:
                rows[case] = row
    return rows


def int_or_none(value):
    if value in (None, ""):
        return None
    return int(float(value))


def discover_h5_cases(h5_dir):
    return sorted(name[:-3] for name in os.listdir(h5_dir) if name.endswith(".h5"))


def has_dataset(h5f, group_name, dataset_name):
    group = h5f.get(group_name)
    if group is not None and dataset_name in group:
        return True
    return dataset_name in h5f


def read_dataset(h5f, group_name, dataset_name):
    group = h5f.get(group_name)
    if group is not None and dataset_name in group:
        return group[dataset_name][:]
    if dataset_name in h5f:
        return h5f[dataset_name][:]
    raise KeyError(dataset_name)


def validate_case(
    case,
    h5_dir,
    manifest_rows,
    label,
    required_features,
    require_ligand_mask,
    exclude_label_atoms_outside_box,
):
    h5_path = os.path.join(h5_dir, f"{case}.h5")
    if not os.path.exists(h5_path):
        return False, "missing_h5"

    manifest_row = manifest_rows.get(case, {})
    if manifest_row.get("status") == "failed":
        return False, "failed_manifest"

    label_prefix = label.replace("binding_site_", "")
    if exclude_label_atoms_outside_box:
        atoms = int_or_none(manifest_row.get(f"{label_prefix}_label_atoms"))
        atoms_in_box = int_or_none(manifest_row.get(f"{label_prefix}_label_atoms_in_box"))
        if atoms is not None and atoms_in_box is not None and atoms_in_box < atoms:
            return False, "label_atoms_outside_box"

    try:
        with h5py.File(h5_path, "r") as h5f:
            if not has_dataset(h5f, "label", label):
                return False, f"missing_label:{label}"

            label_array = read_dataset(h5f, "label", label)
            if int(np.count_nonzero(label_array)) <= 0:
                return False, "empty_label"

            reference_shape = label_array.shape
            for feature_name in required_features:
                if not has_dataset(h5f, "features", feature_name):
                    return False, f"missing_feature:{feature_name}"
                feature_shape = read_dataset(h5f, "features", feature_name).shape
                if feature_shape != reference_shape:
                    return False, f"shape_mismatch:{feature_name}"

            if require_ligand_mask:
                if not has_dataset(h5f, "features", "ligand"):
                    return False, "missing_metric_mask:ligand"
                ligand_shape = read_dataset(h5f, "features", "ligand").shape
                if ligand_shape != reference_shape:
                    return False, "shape_mismatch:ligand"
    except OSError:
        return False, "unreadable_h5"

    return True, "ok"


def write_list(path, values):
    with open(path, "w") as handle:
        for value in values:
            handle.write(f"{value}\n")


def assign_balanced_folds(cases_by_base, fold_count, seed):
    rng = random.Random(seed)
    base_items = [(base_id, sorted(cases)) for base_id, cases in sorted(cases_by_base.items())]
    rng.shuffle(base_items)
    base_items.sort(key=lambda item: len(item[1]), reverse=True)

    folds = [{"base_ids": [], "cases": []} for _ in range(fold_count)]
    for base_id, cases in base_items:
        fold_idx = min(
            range(fold_count),
            key=lambda idx: (len(folds[idx]["cases"]), len(folds[idx]["base_ids"]), idx),
        )
        folds[fold_idx]["base_ids"].append(base_id)
        folds[fold_idx]["cases"].extend(cases)

    for fold in folds:
        fold["base_ids"].sort()
        fold["cases"].sort()
    return folds


def main():
    args = parse_args()
    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")

    required_features = args.required_features or DEFAULT_REQUIRED_FEATURES
    manifest_path = args.manifest or os.path.join(args.h5_dir, "manifest.csv")
    output_dir = args.output_dir or os.path.join(args.h5_dir, f"splits_cache_kfold{args.folds}_seed{args.seed}")
    os.makedirs(output_dir, exist_ok=True)

    manifest_rows = read_manifest_rows(manifest_path)
    discovered_cases = discover_h5_cases(args.h5_dir)
    excluded_rows = []
    valid_cases = []

    for case in discovered_cases:
        ok, reason = validate_case(
            case=case,
            h5_dir=args.h5_dir,
            manifest_rows=manifest_rows,
            label=args.label,
            required_features=required_features,
            require_ligand_mask=not args.allow_missing_ligand_mask,
            exclude_label_atoms_outside_box=not args.allow_label_atoms_outside_box,
        )
        if ok:
            valid_cases.append(case)
        else:
            excluded_rows.append({"case": case, "reason": reason})

    cases_by_base = defaultdict(list)
    for case in valid_cases:
        cases_by_base[strip_scpdb_suffix(case)].append(case)

    folds = assign_balanced_folds(cases_by_base, args.folds, args.seed)
    all_base_ids = set(cases_by_base)

    write_list(os.path.join(output_dir, "valid_cases.txt"), sorted(valid_cases))
    write_list(os.path.join(output_dir, "valid_base_ids.txt"), sorted(all_base_ids))

    with open(os.path.join(output_dir, "excluded_cases.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "reason"])
        writer.writeheader()
        writer.writerows(sorted(excluded_rows, key=lambda row: (row["reason"], row["case"])))

    fold_summaries = []
    for fold_idx, fold in enumerate(folds):
        validation_base_ids = set(fold["base_ids"])
        train_base_ids = all_base_ids - validation_base_ids
        validation_cases = sorted(fold["cases"])
        train_cases = sorted(
            case
            for base_id in train_base_ids
            for case in cases_by_base[base_id]
        )

        write_list(os.path.join(output_dir, f"fold{fold_idx}_train_cases.txt"), train_cases)
        write_list(os.path.join(output_dir, f"fold{fold_idx}_validation_cases.txt"), validation_cases)
        write_list(os.path.join(output_dir, f"fold{fold_idx}_train_base_ids.txt"), sorted(train_base_ids))
        write_list(os.path.join(output_dir, f"fold{fold_idx}_validation_base_ids.txt"), sorted(validation_base_ids))

        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_base_ids": len(train_base_ids),
                "validation_base_ids": len(validation_base_ids),
                "train_cases": len(train_cases),
                "validation_cases": len(validation_cases),
            }
        )

    reason_counts = defaultdict(int)
    for row in excluded_rows:
        reason_counts[row["reason"]] += 1

    with open(os.path.join(output_dir, "summary.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fold", "train_base_ids", "validation_base_ids", "train_cases", "validation_cases"],
        )
        writer.writeheader()
        writer.writerows(fold_summaries)

    with open(os.path.join(output_dir, "summary.json"), "w") as handle:
        json.dump(
            {
                "h5_dir": args.h5_dir,
                "manifest": manifest_path,
                "label": args.label,
                "fold_count": args.folds,
                "seed": args.seed,
                "required_features": required_features,
                "require_ligand_mask": not args.allow_missing_ligand_mask,
                "exclude_label_atoms_outside_box": not args.allow_label_atoms_outside_box,
                "discovered_h5_cases": len(discovered_cases),
                "valid_cases": len(valid_cases),
                "valid_base_ids": len(all_base_ids),
                "excluded_cases": len(excluded_rows),
                "excluded_reasons": dict(sorted(reason_counts.items())),
                "folds": fold_summaries,
            },
            handle,
            indent=2,
        )

    print(f"Output dir: {output_dir}")
    print(f"Discovered H5 cases: {len(discovered_cases)}")
    print(f"Valid cases: {len(valid_cases)}")
    print(f"Valid base IDs: {len(all_base_ids)}")
    print(f"Excluded cases: {len(excluded_rows)}")
    print(f"Excluded reasons: {dict(sorted(reason_counts.items()))}")
    print(f"Summary CSV: {os.path.join(output_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
