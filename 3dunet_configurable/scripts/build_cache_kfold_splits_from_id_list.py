import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from build_cache_kfold_splits import (
    DEFAULT_LABEL,
    DEFAULT_REQUIRED_FEATURES,
    assign_balanced_folds,
    discover_h5_cases,
    strip_scpdb_suffix,
    validate_case,
    write_list,
)
from build_cache_splits_from_fold_ids import read_manifest_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic base-id grouped k-fold splits from an external ID list. "
            "IDs with an scPDB entry suffix such as 1abc_1 are matched exactly; IDs "
            "without a suffix are expanded to all available H5 entries for that base ID."
        )
    )
    parser.add_argument("--h5-dir", required=True)
    parser.add_argument("--id-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
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
        "--h5-exists-only",
        action="store_true",
        help=(
            "Only require that a requested ID resolves to an existing .h5 file. "
            "Do not open H5 files or validate labels/features/metric masks."
        ),
    )
    return parser.parse_args()


def normalize_id(value):
    token = value.strip()
    if not token or token.startswith("#"):
        return ""
    token = re.split(r"[\s,;]+", token, maxsplit=1)[0]
    if token.endswith(".h5"):
        token = token[:-3]
    return token.strip()


def read_id_list(path):
    ids = []
    with open(path) as handle:
        for line in handle:
            token = normalize_id(line)
            if token:
                ids.append(token)
    return ids


def has_entry_suffix(case_id):
    return re.search(r"_[0-9]+$", case_id) is not None


def resolve_requested_cases(requested_ids, h5_cases):
    h5_case_set = set(h5_cases)
    cases_by_base = defaultdict(list)
    for case in h5_cases:
        cases_by_base[strip_scpdb_suffix(case)].append(case)

    selected_cases = set()
    missing_ids = []
    for item in requested_ids:
        if item in h5_case_set:
            selected_cases.add(item)
            continue
        if has_entry_suffix(item):
            missing_ids.append(item)
            continue
        base_id = strip_scpdb_suffix(item)
        if base_id not in cases_by_base:
            missing_ids.append(item)
            continue
        selected_cases.update(cases_by_base[base_id])

    return sorted(selected_cases), sorted(set(missing_ids))


def write_invalid_cases(path, invalid_rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "reason"])
        writer.writeheader()
        writer.writerows(invalid_rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_ids = read_id_list(args.id_list)
    print(f"Requested IDs: {len(requested_ids)}", flush=True)
    h5_cases = discover_h5_cases(args.h5_dir)
    print(f"Indexed available H5 files: {len(h5_cases)} cases", flush=True)
    requested_cases, missing_ids = resolve_requested_cases(requested_ids, h5_cases)
    print(f"Resolved requested H5 cases: {len(requested_cases)}", flush=True)
    print(f"Missing requested IDs: {len(missing_ids)}", flush=True)

    valid_cases = []
    invalid_rows = []
    invalid_reasons = Counter()
    cases_by_base = defaultdict(list)

    if args.h5_exists_only:
        print("Split mode: H5 existence only; skipping H5 content validation", flush=True)
        valid_cases = list(requested_cases)
        for case in valid_cases:
            cases_by_base[strip_scpdb_suffix(case)].append(case)
    else:
        manifest_rows = read_manifest_rows(args.manifest)
        required_features = args.required_features or DEFAULT_REQUIRED_FEATURES
        print(f"Validating requested H5 contents: {len(requested_cases)} cases", flush=True)
        for case_index, case in enumerate(requested_cases, start=1):
            if case_index == 1 or case_index % 500 == 0 or case_index == len(requested_cases):
                print(f"  validated {case_index}/{len(requested_cases)} requested H5 cases", flush=True)
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
                invalid_reasons[status.reason] += 1
                continue
            valid_cases.append(case)
            cases_by_base[strip_scpdb_suffix(case)].append(case)

    for base_id in cases_by_base:
        cases_by_base[base_id].sort()

    all_valid_base_ids = sorted(cases_by_base)
    all_valid_cases = sorted(valid_cases)
    print(f"Available requested base IDs with H5: {len(all_valid_base_ids)}", flush=True)
    print(f"Available requested cases with H5: {len(all_valid_cases)}", flush=True)
    folds = assign_balanced_folds(cases_by_base, args.folds, args.seed)

    write_list(output_dir / "requested_ids.txt", requested_ids)
    write_list(output_dir / "missing_ids.txt", missing_ids)
    write_list(output_dir / "valid_base_ids.txt", all_valid_base_ids)
    write_list(output_dir / "valid_cases.txt", all_valid_cases)
    write_invalid_cases(output_dir / "invalid_cases.csv", invalid_rows)

    summary_rows = []
    for fold_idx, validation_fold in enumerate(folds):
        print(f"Writing fold {fold_idx + 1}/{len(folds)}", flush=True)
        validation_base_ids = sorted(validation_fold["base_ids"])
        validation_cases = sorted(validation_fold["cases"])
        train_base_ids = sorted(
            base_id for base_id in all_valid_base_ids if base_id not in set(validation_base_ids)
        )
        train_cases = sorted(case for base_id in train_base_ids for case in cases_by_base[base_id])

        write_list(output_dir / f"fold{fold_idx}_train_base_ids.txt", train_base_ids)
        write_list(output_dir / f"fold{fold_idx}_validation_base_ids.txt", validation_base_ids)
        write_list(output_dir / f"fold{fold_idx}_train_cases.txt", train_cases)
        write_list(output_dir / f"fold{fold_idx}_validation_cases.txt", validation_cases)
        summary_rows.append(
            {
                "fold": fold_idx,
                "train_base_ids": len(train_base_ids),
                "validation_base_ids": len(validation_base_ids),
                "train_cases": len(train_cases),
                "validation_cases": len(validation_cases),
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
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with (output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "h5_dir": args.h5_dir,
                "id_list": args.id_list,
                "manifest": args.manifest,
                "label": args.label,
                "folds": args.folds,
                "seed": args.seed,
                "split_mode": "h5_exists_only" if args.h5_exists_only else "validated_h5",
                "requested_ids": len(requested_ids),
                "discovered_h5_cases": len(h5_cases),
                "resolved_requested_cases": len(requested_cases),
                "missing_ids": len(missing_ids),
                "valid_cases": len(all_valid_cases),
                "valid_base_ids": len(all_valid_base_ids),
                "invalid_cases": len(invalid_rows),
                "invalid_reasons": dict(sorted(invalid_reasons.items())),
                "fold_summary": summary_rows,
            },
            handle,
            indent=2,
        )

    print(f"Output split dir: {output_dir}")
    print(f"Requested IDs: {len(requested_ids)}")
    print(f"Resolved requested H5 cases: {len(requested_cases)}")
    print(f"Missing IDs: {len(missing_ids)}")
    print(f"Valid cases: {len(all_valid_cases)}")
    print(f"Valid base IDs: {len(all_valid_base_ids)}")
    print(f"Invalid cases: {len(invalid_rows)}")
    print(f"Summary CSV: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
