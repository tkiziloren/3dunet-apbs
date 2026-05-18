#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

from build_cache_kfold_splits import (
    DEFAULT_LABEL,
    DEFAULT_REQUIRED_FEATURES,
    read_manifest_rows,
    validate_case,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check an external case ID list against an H5 cache and write usable, "
            "missing, and invalid ID reports."
        )
    )
    parser.add_argument("--h5-dir", required=True)
    parser.add_argument("--id-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="cache_check")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument(
        "--required-feature",
        action="append",
        dest="required_features",
        default=None,
        help="Feature that must exist in every usable H5. Repeatable.",
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


def write_list(path, values):
    with open(path, "w") as handle:
        for value in values:
            handle.write(f"{value}\n")


def main():
    args = parse_args()
    h5_dir = Path(args.h5_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest or os.path.join(args.h5_dir, "manifest.csv")
    manifest_rows = read_manifest_rows(manifest_path)
    required_features = args.required_features or DEFAULT_REQUIRED_FEATURES
    ids = read_id_list(args.id_list)

    rows = []
    usable = []
    missing_h5 = []
    invalid = []

    for case in ids:
        h5_path = h5_dir / f"{case}.h5"
        if not h5_path.exists():
            rows.append({"case": case, "status": "missing_h5", "reason": str(h5_path)})
            missing_h5.append(case)
            continue

        status = validate_case(
            case=case,
            h5_dir=str(h5_dir),
            manifest_rows=manifest_rows,
            label=args.label,
            required_features=required_features,
            require_ligand_mask=not args.allow_missing_ligand_mask,
            exclude_label_atoms_outside_box=not args.allow_label_atoms_outside_box,
        )
        if status.ok:
            rows.append({"case": case, "status": "usable", "reason": "ok"})
            usable.append(case)
        else:
            rows.append({"case": case, "status": "invalid", "reason": status.reason})
            invalid.append(case)

    report_path = output_dir / f"{args.output_prefix}.csv"
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "status", "reason"])
        writer.writeheader()
        writer.writerows(rows)

    write_list(output_dir / "usable_ids.txt", usable)
    write_list(output_dir / "missing_h5.txt", missing_h5)
    write_list(output_dir / "invalid_ids.txt", invalid)

    invalid_reason_counts = Counter(row["reason"] for row in rows if row["status"] == "invalid")
    summary = {
        "h5_dir": str(h5_dir),
        "id_list": args.id_list,
        "manifest": manifest_path,
        "label": args.label,
        "required_features": required_features,
        "require_ligand_mask": not args.allow_missing_ligand_mask,
        "expected": len(ids),
        "usable": len(usable),
        "missing_h5": len(missing_h5),
        "invalid": len(invalid),
        "invalid_reasons": dict(sorted(invalid_reason_counts.items())),
        "report": str(report_path),
    }
    with (output_dir / f"{args.output_prefix}_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"expected: {len(ids)}")
    print(f"usable: {len(usable)}")
    print(f"missing_h5: {len(missing_h5)}")
    print(f"invalid: {len(invalid)}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
