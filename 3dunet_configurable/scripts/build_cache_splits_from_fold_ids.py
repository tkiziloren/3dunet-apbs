import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from build_cache_kfold_splits import (
    DEFAULT_LABEL,
    DEFAULT_REQUIRED_FEATURES,
    discover_h5_cases,
    strip_scpdb_suffix,
    validate_case,
    write_list,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert published fold ID lists into this repository's cache split files. "
            "Fold IDs are matched against H5 case names and their base IDs; all cases "
            "sharing the same base ID stay in the same split."
        )
    )
    parser.add_argument("--h5-dir", required=True)
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", default="0,1,2,3")
    parser.add_argument(
        "--validation-pattern",
        default="validation_ids_fold{fold}",
        help="Filename pattern under --fold-dir. Use {fold} for the fold index.",
    )
    parser.add_argument(
        "--train-pattern",
        default=None,
        help=(
            "Optional train filename pattern under --fold-dir. If omitted, train IDs are "
            "computed as valid IDs minus validation IDs."
        ),
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--label", default=DEFAULT_LABEL)
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
        help=(
            "Do not require a metric ligand mask. By default ligand is accepted from "
            "features/, auxiliary/, label(s)/, masks/, or the H5 root."
        ),
    )
    parser.add_argument(
        "--allow-missing-fold-ids",
        action="store_true",
        help="Warn and continue when a fold ID is not available in the validated H5 cache.",
    )
    return parser.parse_args()


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


def normalize_id(value):
    token = value.strip()
    if not token or token.startswith("#"):
        return ""
    token = re.split(r"[\s,;]+", token, maxsplit=1)[0]
    if token.endswith(".h5"):
        token = token[:-3]
    return token.strip()


def read_id_file(path):
    ids = []
    with open(path) as handle:
        for line in handle:
            token = normalize_id(line)
            if token:
                ids.append(token)
    return ids


def fold_path(fold_dir, pattern, fold):
    return Path(fold_dir) / pattern.format(fold=fold)


def case_to_base_id(case):
    return strip_scpdb_suffix(case)


def build_valid_case_index(args):
    manifest_rows = read_manifest_rows(args.manifest)
    required_features = args.required_features or DEFAULT_REQUIRED_FEATURES
    cases_by_base = defaultdict(list)
    invalid_rows = []

    for case in discover_h5_cases(args.h5_dir):
        status = validate_case(
            case,
            args.h5_dir,
            manifest_rows,
            args.label,
            required_features,
            require_ligand_mask=not args.allow_missing_ligand_mask,
            exclude_label_atoms_outside_box=not args.allow_label_atoms_outside_box,
        )
        if not status.ok:
            invalid_rows.append({"case": case, "reason": status.reason})
            continue
        cases_by_base[case_to_base_id(case)].append(case)

    for base_id in cases_by_base:
        cases_by_base[base_id].sort()
    return dict(cases_by_base), invalid_rows


def ids_to_base_ids(ids, available_base_ids):
    base_ids = []
    missing = []
    for item in ids:
        base_id = case_to_base_id(item)
        if item in available_base_ids:
            base_id = item
        elif base_id not in available_base_ids:
            missing.append(item)
            continue
        base_ids.append(base_id)
    return sorted(set(base_ids)), missing


def main():
    args = parse_args()
    folds = [int(item.strip()) for item in args.folds.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases_by_base, invalid_rows = build_valid_case_index(args)
    all_base_ids = set(cases_by_base)
    write_list(output_dir / "valid_base_ids.txt", sorted(all_base_ids))

    if invalid_rows:
        with (output_dir / "invalid_cases.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["case", "reason"])
            writer.writeheader()
            writer.writerows(invalid_rows)

    summary_rows = []
    missing_by_fold = {}

    for fold in folds:
        validation_path = fold_path(args.fold_dir, args.validation_pattern, fold)
        if not validation_path.exists():
            raise SystemExit(f"Missing validation fold file: {validation_path}")

        validation_ids, missing_validation = ids_to_base_ids(read_id_file(validation_path), all_base_ids)

        if args.train_pattern:
            train_path = fold_path(args.fold_dir, args.train_pattern, fold)
            if not train_path.exists():
                raise SystemExit(f"Missing train fold file: {train_path}")
            train_ids, missing_train = ids_to_base_ids(read_id_file(train_path), all_base_ids)
        else:
            train_ids = sorted(all_base_ids.difference(validation_ids))
            missing_train = []

        overlap = set(train_ids).intersection(validation_ids)
        if overlap:
            raise SystemExit(f"Fold {fold} has {len(overlap)} base IDs in both train and validation.")

        missing = missing_train + missing_validation
        if missing:
            missing_by_fold[str(fold)] = sorted(missing)
            if not args.allow_missing_fold_ids:
                raise SystemExit(
                    f"Fold {fold} has {len(missing)} IDs not present in validated H5 cache. "
                    f"Re-run with --allow-missing-fold-ids to continue."
                )

        train_cases = sorted(case for base_id in train_ids for case in cases_by_base[base_id])
        validation_cases = sorted(case for base_id in validation_ids for case in cases_by_base[base_id])

        write_list(output_dir / f"fold{fold}_train_base_ids.txt", train_ids)
        write_list(output_dir / f"fold{fold}_validation_base_ids.txt", validation_ids)
        write_list(output_dir / f"fold{fold}_train_cases.txt", train_cases)
        write_list(output_dir / f"fold{fold}_validation_cases.txt", validation_cases)
        summary_rows.append(
            {
                "fold": fold,
                "train_base_ids": len(train_ids),
                "validation_base_ids": len(validation_ids),
                "train_cases": len(train_cases),
                "validation_cases": len(validation_cases),
                "missing_fold_ids": len(missing),
            }
        )

    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "train_base_ids",
                "validation_base_ids",
                "train_cases",
                "validation_cases",
                "missing_fold_ids",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with (output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "h5_dir": args.h5_dir,
                "fold_dir": args.fold_dir,
                "folds": folds,
                "valid_base_ids": len(all_base_ids),
                "invalid_cases": len(invalid_rows),
                "missing_by_fold": missing_by_fold,
            },
            handle,
            indent=2,
        )

    print(f"Output split dir: {output_dir}")
    print(f"Valid base IDs: {len(all_base_ids)}")
    print(f"Invalid cases: {len(invalid_rows)}")
    print(f"Summary CSV: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
