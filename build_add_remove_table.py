#!/usr/bin/env python3
"""
Build CSV of unique Voter IDs added/removed by senate district between consecutive scrape dates.

Reads HTML/XLSX snapshots from HTMLs/ (one file per date), computes day-over-day deltas per
district, and writes a fully rebuilt signature_changes_by_district.csv.

Output guarantees one row per district for every date entry:
- Initial rows: (From_Date="", To_Date=<first_date>) for every district.
- Transition rows: one row per district for each consecutive (From_Date, To_Date),
  including 0/0 rows when no voter IDs changed in that district.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# Optional for type hints only
try:
    from lxml import etree
    from lxml import html as lxml_html
except ImportError:
    etree = None  # type: ignore[assignment]
    lxml_html = None  # type: ignore[assignment]


CSV_COLUMNS = ("From_Date", "To_Date", "Senate_District", "Added", "Removed")
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})\.(html|xlsx)$", re.IGNORECASE)


class Snapshot(NamedTuple):
    """Per-date snapshot: set of voter IDs and id -> senate district."""

    ids: set[str]
    id_to_district: dict[str, str]


class ParseStats(NamedTuple):
    """Counts for data discrepancy reporting."""

    malformed_rows: int
    duplicate_ids: int


def discover_snapshot_files(html_dir: Path) -> dict[str, Path]:
    """
    Discover YYYY-MM-DD.(html|xlsx) files in html_dir.

    Returns mapping of date string -> Path. If both HTML and XLSX exist
    for a given date, XLSX is preferred.
    """
    snapshots: dict[str, Path] = {}
    for p in html_dir.iterdir():
        if not p.is_file():
            continue
        m = DATE_PATTERN.search(p.name)
        if not m:
            continue
        date = m.group(1)
        existing = snapshots.get(date)
        if existing is None:
            snapshots[date] = p
        else:
            # Prefer XLSX over HTML if both are present
            if existing.suffix.lower() == ".html" and p.suffix.lower() == ".xlsx":
                snapshots[date] = p
    return snapshots


def parse_signers_table_html(filepath: Path) -> tuple[Snapshot, ParseStats]:
    """
    Parse the signer table from one HTML file; return snapshot and discrepancy counts.

    Uses lxml for memory-efficient parsing. Extracts Voter ID (1st column) and
    Senate District (4th column). Deduplicates by Voter ID (last occurrence wins).
    """
    if lxml_html is None or etree is None:
        raise RuntimeError("lxml is required. Install with: pip install lxml")

    malformed = 0
    seen_ids: set[str] = set()
    duplicate_count = 0
    id_to_district: dict[str, str] = {}
    ids_set: set[str] = set()

    doc = lxml_html.parse(str(filepath))
    # Find table that has "Voter ID" and "Senate District" headers
    for table in doc.iter("table"):
        headers = [th.text_content().strip() if th.text is not None else "" for th in table.findall(".//th")]
        if "Voter ID" in headers and "Senate District" in headers:
            rows = table.findall(".//tr")
            for tr in rows:
                tds = tr.findall("td")
                if len(tds) != 4:
                    if len(tds) > 0:  # header row has th, not td; skip non-data
                        malformed += 1
                    continue
                voter_id = (tds[0].text or "").strip()
                district = (tds[3].text or "").strip()
                if not voter_id:
                    malformed += 1
                    continue
                if voter_id in seen_ids:
                    duplicate_count += 1
                seen_ids.add(voter_id)
                id_to_district[voter_id] = district
                ids_set.add(voter_id)
            break
    else:
        # No matching table found
        pass

    return Snapshot(ids=ids_set, id_to_district=id_to_district), ParseStats(
        malformed_rows=malformed, duplicate_ids=duplicate_count
    )


def parse_signers_table_excel(filepath: Path) -> tuple[Snapshot, ParseStats]:
    malformed = 0
    seen_ids: set[str] = set()
    duplicate_count = 0
    id_to_district: dict[str, str] = {}
    ids_set: set[str] = set()

    df = pd.read_excel(filepath)

    normalized_columns = {str(c).strip().lower(): c for c in df.columns}
    try:
        voter_col = normalized_columns["voter id"]
        district_col = normalized_columns["senate district"]
    except KeyError as exc:
        raise RuntimeError(
            f"Expected 'Voter ID' and 'Senate District' columns in {filepath.name}"
        ) from exc

    for _, row in df.iterrows():
        voter_val = row[voter_col]
        district_val = row[district_col]

        voter_id = str(voter_val).strip() if pd.notna(voter_val) else ""
        district = str(district_val).strip() if pd.notna(district_val) else ""

        if not voter_id:
            malformed += 1
            continue

        if voter_id in seen_ids:
            duplicate_count += 1
        seen_ids.add(voter_id)

        id_to_district[voter_id] = district
        ids_set.add(voter_id)

    return Snapshot(ids=ids_set, id_to_district=id_to_district), ParseStats(
        malformed_rows=malformed, duplicate_ids=duplicate_count
    )


def parse_signers_table(filepath: Path) -> tuple[Snapshot, ParseStats]:
    suffix = filepath.suffix.lower()
    if suffix == ".xlsx":
        return parse_signers_table_excel(filepath)
    if suffix in {".html", ".htm"}:
        return parse_signers_table_html(filepath)
    raise RuntimeError(f"Unsupported snapshot file type: {filepath}")


def compute_initial_rows(
    first_date: str,
    snapshot: Snapshot,
    all_districts: list[str],
) -> list[tuple[str, str, str, int, int]]:
    """
    Compute initial rows showing signature counts per district on the first date.
    
    Returns list of ("", first_date, Senate_District, Added=count, Removed=0)
    for every district in all_districts.
    """
    district_counts: dict[str, int] = {}
    for vid, district in snapshot.id_to_district.items():
        if district:
            district_counts[district] = district_counts.get(district, 0) + 1
    
    rows: list[tuple[str, str, str, int, int]] = []
    for district in all_districts:
        rows.append(("", first_date, district, district_counts.get(district, 0), 0))
    return rows


def compute_transition(
    prev: Snapshot,
    curr: Snapshot,
) -> tuple[dict[str, int], dict[str, int], int]:
    """
    Compute added/removed counts per district for one date transition.

    Returns (added_by_district, removed_by_district, district_change_count),
    where district_change_count is the number of voter IDs that appeared on both
    dates with different district values.
    """
    prev_ids = prev.ids
    curr_ids = curr.ids
    added_ids = curr_ids - prev_ids
    removed_ids = prev_ids - curr_ids

    # Added: attribute by district from curr snapshot
    added_by_district: dict[str, int] = {}
    for vid in added_ids:
        d = curr.id_to_district.get(vid, "")
        if d:
            added_by_district[d] = added_by_district.get(d, 0) + 1

    # Removed: attribute by district from prev snapshot
    removed_by_district: dict[str, int] = {}
    for vid in removed_ids:
        d = prev.id_to_district.get(vid, "")
        if d:
            removed_by_district[d] = removed_by_district.get(d, 0) + 1

    district_changes = 0
    both = prev_ids & curr_ids
    for vid in both:
        if prev.id_to_district.get(vid) != curr.id_to_district.get(vid):
            district_changes += 1

    return added_by_district, removed_by_district, district_changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("signature_changes_by_district.csv"), help="Output CSV path")
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=Path("HTMLs"),
        help="Directory containing YYYY-MM-DD HTML/XLSX snapshot files",
    )
    args = parser.parse_args()

    html_dir = args.html_dir.resolve()
    csv_path = args.csv.resolve()

    if not html_dir.is_dir():
        print(f"Error: HTML directory not found: {html_dir}", file=sys.stderr)
        return 1

    date_to_path = discover_snapshot_files(html_dir)
    dates = sorted(date_to_path.keys())
    if len(dates) == 0:
        print("No date files found.", file=sys.stderr)
        return 0
    
    first_date = dates[0]

    # Collect discrepancy messages
    malformed_total = 0
    duplicate_total = 0
    district_change_total = 0

    first_path = date_to_path.get(first_date)
    if first_path is None or not first_path.exists():
        print(f"Error: Missing file for first date {first_date}", file=sys.stderr)
        return 1

    first_snap, first_stats = parse_signers_table(first_path)
    malformed_total += first_stats.malformed_rows
    duplicate_total += first_stats.duplicate_ids
    if first_stats.malformed_rows:
        print(f"Skipped {first_stats.malformed_rows} malformed row(s) in {first_path.name}.", file=sys.stderr)
    if first_stats.duplicate_ids:
        print(f"Deduplicated {first_stats.duplicate_ids} duplicate Voter ID(s) in {first_path.name}.", file=sys.stderr)

    prev_snap = first_snap

    district_universe: set[str] = {d for d in prev_snap.id_to_district.values() if d}
    transition_results: list[tuple[str, str, dict[str, int], dict[str, int]]] = []

    for idx in range(1, len(dates)):
        from_date = dates[idx - 1]
        to_date = dates[idx]
        curr_path = date_to_path.get(to_date)
        if curr_path is None or not curr_path.exists():
            print(f"Warning: Missing file for transition {from_date} -> {to_date}", file=sys.stderr)
            continue

        curr_snap, curr_stats = parse_signers_table(curr_path)
        malformed_total += curr_stats.malformed_rows
        duplicate_total += curr_stats.duplicate_ids
        if curr_stats.malformed_rows:
            print(f"Skipped {curr_stats.malformed_rows} malformed row(s) in {curr_path.name}.", file=sys.stderr)
        if curr_stats.duplicate_ids:
            print(f"Deduplicated {curr_stats.duplicate_ids} duplicate Voter ID(s) in {curr_path.name}.", file=sys.stderr)

        district_universe.update(d for d in curr_snap.id_to_district.values() if d)

        added_by_district, removed_by_district, district_changes = compute_transition(prev_snap, curr_snap)
        district_change_total += district_changes

        transition_results.append((from_date, to_date, added_by_district, removed_by_district))
        prev_snap = curr_snap

    if district_change_total:
        print(f"{district_change_total} Voter ID(s) had different senate district on consecutive dates.", file=sys.stderr)

    all_districts = sorted(district_universe)
    initial_rows = compute_initial_rows(first_date, first_snap, all_districts)

    transition_rows: list[tuple[str, str, str, int, int]] = []
    for from_date, to_date, added_by_district, removed_by_district in transition_results:
        for district in all_districts:
            transition_rows.append(
                (
                    from_date,
                    to_date,
                    district,
                    added_by_district.get(district, 0),
                    removed_by_district.get(district, 0),
                )
            )

    all_rows_to_write = initial_rows + transition_rows

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(all_rows_to_write)

    print(
        f"Wrote {len(initial_rows)} initial row(s) and {len(transition_rows)} transition row(s) to {csv_path}."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
