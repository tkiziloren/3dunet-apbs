import argparse
import csv
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SCPDB_RESULTS_URL = "http://bioinfo-pharma.u-strasbg.fr/scPDB/RESULTS"
INTERPRO_PFAM_URL = "https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/uniprot/{accession}/?page_size=100"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a case-to-family/group CSV for scPDB/PUResNet IDs. The script queries "
            "scPDB for UniProt accessions, then InterPro/Pfam for family/domain signatures."
        )
    )
    parser.add_argument("--id-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--group-prefix",
        default="PFAM",
        help="Prefix added to Pfam-derived groups, for example PFAM:PF00069.",
    )
    parser.add_argument(
        "--fallback-to-uniprot",
        action="store_true",
        help="Use UNIPROT:<accession> when no Pfam signature is found.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-query remote services even when cached responses are available.",
    )
    return parser.parse_args()


def request_text(url, data=None, timeout=30.0):
    body = None if data is None else urlencode(data).encode()
    request = Request(url, data=body, headers={"User-Agent": "deep-apbs-family-map/1.0"})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def request_json(url, timeout=30.0):
    request = Request(url, headers={"User-Agent": "deep-apbs-family-map/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def cached_text(path, fetch, force=False):
    if path.exists() and not force:
        return path.read_text(errors="replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = fetch()
    path.write_text(text)
    return text


def cached_json(path, fetch, force=False):
    if path.exists() and not force:
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = fetch()
    path.write_text(json.dumps(payload, indent=2))
    return payload


def strip_tags(value):
    value = re.sub(r"<[^>]+>", "", value or "")
    return html.unescape(value).strip()


def strip_scpdb_suffix(case_id):
    return re.sub(r"_[0-9]+$", "", case_id)


def normalize_id(value):
    token = (value or "").strip()
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


def extract_table_value(page, label):
    pattern = rf"<th[^>]*>\s*{re.escape(label)}\s*:\s*</th>\s*<td[^>]*>(.*?)</td>"
    match = re.search(pattern, page, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return strip_tags(match.group(1))


def parse_scpdb_page(page):
    entry_match = re.search(r"ressources/2016/entries/([^/]+)/+protein\.mol2", page)
    return {
        "scpdb_entry_case": entry_match.group(1) if entry_match else "",
        "protein_name": extract_table_value(page, "Name"),
        "uniprot_id": extract_table_value(page, "ID"),
        "uniprot_ac": extract_table_value(page, "AC"),
        "organism": extract_table_value(page, "Organism"),
        "tax_id": extract_table_value(page, "TaxID"),
        "ec_number": extract_table_value(page, "EC Number"),
    }


def score_pfam_result(result):
    proteins = result.get("proteins") or []
    total_coverage = 0
    best_score = None
    for protein in proteins:
        for location in protein.get("entry_protein_locations") or []:
            score = location.get("score")
            if score is not None and (best_score is None or score < best_score):
                best_score = score
            for fragment in location.get("fragments") or []:
                start = fragment.get("start")
                end = fragment.get("end")
                if isinstance(start, int) and isinstance(end, int) and end >= start:
                    total_coverage += end - start + 1
    return total_coverage, best_score if best_score is not None else float("inf")


def choose_pfam_family(payload):
    results = payload.get("results") or []
    if not results:
        return {}

    ranked = []
    for result in results:
        metadata = result.get("metadata") or {}
        accession = metadata.get("accession") or ""
        if not accession:
            continue
        coverage, score = score_pfam_result(result)
        ranked.append(
            {
                "accession": accession,
                "name": metadata.get("name") or "",
                "type": metadata.get("type") or "",
                "integrated": metadata.get("integrated") or "",
                "coverage": coverage,
                "score": score,
            }
        )
    if not ranked:
        return {}

    family_like = [item for item in ranked if item["type"] == "family"]
    candidates = family_like or ranked
    candidates.sort(key=lambda item: (-item["coverage"], item["score"], item["accession"]))
    return candidates[0]


def existing_completed_rows(path):
    if not path.exists():
        return {}
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("case") and row.get("status") == "ok":
                rows[row["case"]] = row
    return rows


def main():
    args = parse_args()
    output_path = Path(args.output)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_path.with_suffix(output_path.suffix + ".cache")
    scpdb_cache = cache_dir / "scpdb"
    interpro_cache = cache_dir / "interpro_pfam"

    ids = read_id_list(args.id_list)
    if args.limit:
        ids = ids[: args.limit]
    completed = existing_completed_rows(output_path)

    fieldnames = [
        "case",
        "base_id",
        "group",
        "group_source",
        "group_name",
        "scpdb_entry_case",
        "uniprot_ac",
        "uniprot_id",
        "protein_name",
        "organism",
        "tax_id",
        "ec_number",
        "pfam_accession",
        "pfam_name",
        "pfam_type",
        "pfam_integrated",
        "pfam_coverage",
        "pfam_score",
        "status",
        "error",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    ok_count = 0
    fail_count = 0

    for index, case in enumerate(ids, start=1):
        if case in completed and not args.force:
            rows.append(completed[case])
            ok_count += 1
            continue

        base_id = strip_scpdb_suffix(case)
        row = {name: "" for name in fieldnames}
        row["case"] = case
        row["base_id"] = base_id

        try:
            if index == 1 or index % 50 == 0 or index == len(ids):
                print(f"Processing {index}/{len(ids)}: {case}", flush=True)

            scpdb_page = cached_text(
                scpdb_cache / f"{base_id}.html",
                lambda: request_text(SCPDB_RESULTS_URL, data={"PDB_ID": base_id}, timeout=args.timeout),
                force=args.force,
            )
            annotation = parse_scpdb_page(scpdb_page)
            row.update(annotation)
            accession = annotation.get("uniprot_ac") or ""
            if not accession:
                raise RuntimeError("missing_uniprot_ac")

            pfam_payload = cached_json(
                interpro_cache / f"{accession}.json",
                lambda: request_json(INTERPRO_PFAM_URL.format(accession=accession), timeout=args.timeout),
                force=args.force,
            )
            pfam = choose_pfam_family(pfam_payload)
            if pfam:
                row["pfam_accession"] = pfam["accession"]
                row["pfam_name"] = pfam["name"]
                row["pfam_type"] = pfam["type"]
                row["pfam_integrated"] = pfam["integrated"]
                row["pfam_coverage"] = pfam["coverage"]
                row["pfam_score"] = pfam["score"]
                row["group"] = f"{args.group_prefix}:{pfam['accession']}"
                row["group_source"] = "pfam"
                row["group_name"] = pfam["name"]
            elif args.fallback_to_uniprot:
                row["group"] = f"UNIPROT:{accession}"
                row["group_source"] = "uniprot_fallback"
                row["group_name"] = accession
            else:
                raise RuntimeError("missing_pfam_group")

            row["status"] = "ok"
            ok_count += 1
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}:{exc}"
            fail_count += 1

        rows.append(row)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Output: {output_path}")
    print(f"Rows: {len(rows)}")
    print(f"OK: {ok_count}")
    print(f"Failed: {fail_count}")
    print(f"Cache dir: {cache_dir}")


if __name__ == "__main__":
    main()
