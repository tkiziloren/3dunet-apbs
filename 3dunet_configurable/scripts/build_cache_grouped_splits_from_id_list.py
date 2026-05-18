import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from build_cache_kfold_splits import (
    DEFAULT_LABEL,
    DEFAULT_REQUIRED_FEATURES,
    discover_h5_cases,
    strip_scpdb_suffix,
    validate_case,
    write_list,
)
from build_cache_kfold_splits_from_id_list import read_id_list, resolve_requested_cases
from build_cache_splits_from_fold_ids import read_manifest_rows


ID_COLUMN_CANDIDATES = ("case", "case_id", "id", "base_id", "pdb_id", "scpdb_id")
GROUP_COLUMN_CANDIDATES = (
    "group",
    "family",
    "protein_family",
    "uniprot_family",
    "uniprot_ac",
    "uniprot_id",
    "cluster",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build group-aware k-fold splits from an external ID list and a case/base-id "
            "to family/group map. All base IDs assigned to the same group stay in the "
            "same validation fold."
        )
    )
    parser.add_argument("--h5-dir", required=True)
    parser.add_argument("--id-list", required=True)
    parser.add_argument("--group-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--group-column", default=None)
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
    parser.add_argument(
        "--allow-missing-groups",
        action="store_true",
        help=(
            "Allow requested H5 cases without a group-map entry by assigning each missing "
            "base ID to its own singleton group. This is not paper-faithful."
        ),
    )
    return parser.parse_args()


def normalize_id(value):
    token = (value or "").strip()
    if not token or token.startswith("#"):
        return ""
    token = re.split(r"[\s,;]+", token, maxsplit=1)[0]
    if token.endswith(".h5"):
        token = token[:-3]
    return token.strip()


def normalize_group(value):
    return re.sub(r"\s+", " ", (value or "").strip())


def sniff_dialect(path):
    with open(path, newline="") as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        return csv.excel


def choose_column(fieldnames, requested, candidates, label):
    if requested:
        if requested not in fieldnames:
            raise SystemExit(f"Requested {label} column '{requested}' not found in group map.")
        return requested
    lower_to_original = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    raise SystemExit(
        f"Could not infer {label} column from {fieldnames}. "
        f"Use --{label.replace('_', '-')}-column."
    )


def read_group_map(path, id_column=None, group_column=None):
    dialect = sniff_dialect(path)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise SystemExit(f"Group map has no header row: {path}")
        id_col = choose_column(reader.fieldnames, id_column, ID_COLUMN_CANDIDATES, "id")
        group_col = choose_column(reader.fieldnames, group_column, GROUP_COLUMN_CANDIDATES, "group")

        group_by_id = {}
        conflicts = []
        rows_used = []
        for row in reader:
            raw_id = normalize_id(row.get(id_col))
            group = normalize_group(row.get(group_col))
            if not raw_id or not group:
                continue
            keys = {raw_id, strip_scpdb_suffix(raw_id)}
            for key in keys:
                previous = group_by_id.get(key)
                if previous is not None and previous != group:
                    conflicts.append({"id": key, "previous_group": previous, "group": group})
                    continue
                group_by_id[key] = group
            rows_used.append({"id": raw_id, "base_id": strip_scpdb_suffix(raw_id), "group": group})

    if conflicts:
        preview = ", ".join(f"{item['id']}:{item['previous_group']}!={item['group']}" for item in conflicts[:5])
        raise SystemExit(f"Conflicting group assignments in {path}: {preview}")
    return group_by_id, rows_used, id_col, group_col


def write_invalid_cases(path, invalid_rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "reason"])
        writer.writeheader()
        writer.writerows(invalid_rows)


def build_valid_cases(args, requested_cases):
    valid_cases = []
    invalid_rows = []
    invalid_reasons = Counter()
    cases_by_base = defaultdict(list)

    if args.h5_exists_only:
        print("Split mode: H5 existence only; skipping H5 content validation", flush=True)
        valid_cases = list(requested_cases)
        for case in valid_cases:
            cases_by_base[strip_scpdb_suffix(case)].append(case)
        return valid_cases, cases_by_base, invalid_rows, invalid_reasons

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

    return valid_cases, cases_by_base, invalid_rows, invalid_reasons


def assign_grouped_folds(group_to_base_ids, cases_by_base, fold_count, seed):
    rng = random.Random(seed)
    group_items = []
    for group, base_ids in sorted(group_to_base_ids.items()):
        sorted_base_ids = sorted(base_ids)
        cases = sorted(case for base_id in sorted_base_ids for case in cases_by_base[base_id])
        group_items.append({"group": group, "base_ids": sorted_base_ids, "cases": cases})

    rng.shuffle(group_items)
    group_items.sort(key=lambda item: (len(item["cases"]), len(item["base_ids"])), reverse=True)

    folds = [{"groups": [], "base_ids": [], "cases": []} for _ in range(fold_count)]
    for item in group_items:
        fold_idx = min(
            range(fold_count),
            key=lambda idx: (
                len(folds[idx]["cases"]),
                len(folds[idx]["base_ids"]),
                len(folds[idx]["groups"]),
                idx,
            ),
        )
        folds[fold_idx]["groups"].append(item["group"])
        folds[fold_idx]["base_ids"].extend(item["base_ids"])
        folds[fold_idx]["cases"].extend(item["cases"])

    for fold in folds:
        fold["groups"].sort()
        fold["base_ids"].sort()
        fold["cases"].sort()
    return folds


def main():
    args = parse_args()
    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_ids = read_id_list(args.id_list)
    print(f"Requested IDs: {len(requested_ids)}", flush=True)
    h5_cases = discover_h5_cases(args.h5_dir)
    print(f"Indexed available H5 files: {len(h5_cases)} cases", flush=True)
    requested_cases, missing_ids = resolve_requested_cases(requested_ids, h5_cases)
    print(f"Resolved requested H5 cases: {len(requested_cases)}", flush=True)
    print(f"Missing requested IDs: {len(missing_ids)}", flush=True)

    valid_cases, cases_by_base, invalid_rows, invalid_reasons = build_valid_cases(args, requested_cases)
    for base_id in cases_by_base:
        cases_by_base[base_id].sort()

    all_valid_base_ids = sorted(cases_by_base)
    all_valid_cases = sorted(valid_cases)
    print(f"Available requested base IDs with H5: {len(all_valid_base_ids)}", flush=True)
    print(f"Available requested cases with H5: {len(all_valid_cases)}", flush=True)

    group_by_id, group_rows, id_col, group_col = read_group_map(
        args.group_map,
        id_column=args.id_column,
        group_column=args.group_column,
    )
    print(f"Read group map rows: {len(group_rows)}", flush=True)
    print(f"Group map columns: id={id_col} group={group_col}", flush=True)

    base_to_group = {}
    group_to_base_ids = defaultdict(set)
    missing_group_base_ids = []
    for base_id, cases in sorted(cases_by_base.items()):
        group = group_by_id.get(base_id)
        if group is None:
            for case in cases:
                group = group_by_id.get(case)
                if group is not None:
                    break
        if group is None:
            missing_group_base_ids.append(base_id)
            if not args.allow_missing_groups:
                continue
            group = f"__ungrouped__:{base_id}"
        base_to_group[base_id] = group
        group_to_base_ids[group].add(base_id)

    write_list(output_dir / "missing_group_ids.txt", sorted(missing_group_base_ids))
    if missing_group_base_ids and not args.allow_missing_groups:
        raise SystemExit(
            f"{len(missing_group_base_ids)} valid base IDs have no group-map entry. "
            "Add them to --group-map or re-run with --allow-missing-groups."
        )

    print(f"Grouped base IDs: {len(base_to_group)}", flush=True)
    print(f"Groups: {len(group_to_base_ids)}", flush=True)
    folds = assign_grouped_folds(group_to_base_ids, cases_by_base, args.folds, args.seed)

    write_list(output_dir / "requested_ids.txt", requested_ids)
    write_list(output_dir / "missing_ids.txt", missing_ids)
    write_list(output_dir / "valid_base_ids.txt", all_valid_base_ids)
    write_list(output_dir / "valid_cases.txt", all_valid_cases)
    write_invalid_cases(output_dir / "invalid_cases.csv", invalid_rows)
    with (output_dir / "group_map_used.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["base_id", "group"])
        writer.writeheader()
        for base_id in sorted(base_to_group):
            writer.writerow({"base_id": base_id, "group": base_to_group[base_id]})

    summary_rows = []
    for fold_idx, validation_fold in enumerate(folds):
        print(f"Writing fold {fold_idx + 1}/{len(folds)}", flush=True)
        validation_groups = sorted(validation_fold["groups"])
        validation_base_ids = sorted(validation_fold["base_ids"])
        validation_cases = sorted(validation_fold["cases"])
        validation_base_set = set(validation_base_ids)
        train_base_ids = sorted(base_id for base_id in all_valid_base_ids if base_id not in validation_base_set)
        train_cases = sorted(case for base_id in train_base_ids for case in cases_by_base[base_id])
        train_groups = sorted({base_to_group[base_id] for base_id in train_base_ids})

        overlap_groups = set(train_groups).intersection(validation_groups)
        if overlap_groups:
            preview = ", ".join(sorted(overlap_groups)[:10])
            raise SystemExit(f"Fold {fold_idx} leaks groups across train/validation: {preview}")

        write_list(output_dir / f"fold{fold_idx}_train_groups.txt", train_groups)
        write_list(output_dir / f"fold{fold_idx}_validation_groups.txt", validation_groups)
        write_list(output_dir / f"fold{fold_idx}_train_base_ids.txt", train_base_ids)
        write_list(output_dir / f"fold{fold_idx}_validation_base_ids.txt", validation_base_ids)
        write_list(output_dir / f"fold{fold_idx}_train_cases.txt", train_cases)
        write_list(output_dir / f"fold{fold_idx}_validation_cases.txt", validation_cases)
        summary_rows.append(
            {
                "fold": fold_idx,
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
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
                "train_groups",
                "validation_groups",
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
                "group_map": args.group_map,
                "id_column": id_col,
                "group_column": group_col,
                "manifest": args.manifest,
                "label": args.label,
                "folds": args.folds,
                "seed": args.seed,
                "split_mode": "grouped_h5_exists_only" if args.h5_exists_only else "grouped_validated_h5",
                "split_policy": (
                    "requested IDs filtered to available H5 cases; k-fold assignment keeps "
                    "all base IDs from the same supplied group in the same validation fold"
                ),
                "requested_ids": len(requested_ids),
                "discovered_h5_cases": len(h5_cases),
                "resolved_requested_cases": len(requested_cases),
                "missing_ids": len(missing_ids),
                "valid_cases": len(all_valid_cases),
                "valid_base_ids": len(all_valid_base_ids),
                "groups": len(group_to_base_ids),
                "missing_group_ids": len(missing_group_base_ids),
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
    print(f"Groups: {len(group_to_base_ids)}")
    print(f"Missing group IDs: {len(missing_group_base_ids)}")
    print(f"Invalid cases: {len(invalid_rows)}")
    print(f"Summary CSV: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
