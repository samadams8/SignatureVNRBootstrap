#!/usr/bin/env python3
"""
Build CSV of unique Voter IDs added/removed by senate district between consecutive scrape dates.

Reads HTML snapshots from HTMLs/ (one file per date), computes day-over-day deltas per
district, and writes or appends to signature_changes_by_district.csv. Re-runs only
process date transitions not already present in the CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import NamedTuple

# Optional for type hints only
try:
    from lxml import etree
    from lxml import html as lxml_html
except ImportError:
    etree = None  # type: ignore[assignment]
    lxml_html = None  # type: ignore[assignment]


CSV_COLUMNS = ("From_Date", "To_Date", "Senate_District", "Added", "Removed")
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})\.html$")


class Snapshot(NamedTuple):
    """Per-date snapshot: set of voter IDs and id -> senate district."""

    ids: set[str]
    id_to_district: dict[str, str]


class ParseStats(NamedTuple):
    """Counts for data discrepancy reporting."""

    malformed_rows: int
    duplicate_ids: int


def get_sorted_dates(html_dir: Path) -> list[str]:
    """Discover YYYY-MM-DD.html files in html_dir and return sorted date strings."""
    dates: list[str] = []
    for p in html_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".html":
            continue
        m = DATE_PATTERN.search(p.name)
        if m:
            dates.append(m.group(1))
    dates.sort()
    return dates


def parse_signers_table(filepath: Path) -> tuple[Snapshot, ParseStats]:
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


def compute_initial_rows(first_date: str, snapshot: Snapshot) -> list[tuple[str, str, str, int, int]]:
    """
    Compute initial rows showing signature counts per district on the first date.
    
    Returns list of ("", first_date, Senate_District, Added=count, Removed=0).
    """
    district_counts: dict[str, int] = {}
    for vid, district in snapshot.id_to_district.items():
        if district:
            district_counts[district] = district_counts.get(district, 0) + 1
    
    rows: list[tuple[str, str, str, int, int]] = []
    for district in sorted(district_counts.keys()):
        rows.append(("", first_date, district, district_counts[district], 0))
    return rows


def compute_transition(
    prev_date: str,
    curr_date: str,
    prev: Snapshot,
    curr: Snapshot,
) -> tuple[list[tuple[str, str, str, int, int]], int]:
    """
    Compute added/removed counts per district for one date transition.

    Returns list of (From_Date, To_Date, Senate_District, Added, Removed) and
    count of voter IDs that appeared on both dates with different district.
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

    # All districts that had any add or remove
    all_districts = sorted(set(added_by_district) | set(removed_by_district))

    district_changes = 0
    both = prev_ids & curr_ids
    for vid in both:
        if prev.id_to_district.get(vid) != curr.id_to_district.get(vid):
            district_changes += 1

    rows: list[tuple[str, str, str, int, int]] = []
    for d in all_districts:
        rows.append((
            prev_date,
            curr_date,
            d,
            added_by_district.get(d, 0),
            removed_by_district.get(d, 0),
        ))
    return rows, district_changes


def read_existing_transitions(csv_path: Path) -> tuple[set[tuple[str, str]], bool]:
    """
    Stream CSV and return set of (From_Date, To_Date) already present, and whether initial rows exist.
    
    Returns (existing_transitions_set, has_initial_rows).
    Initial rows are those with empty From_Date.
    """
    existing: set[tuple[str, str]] = set()
    has_initial = False
    if not csv_path.exists():
        return existing, has_initial
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and reader.fieldnames[:2] != list(CSV_COLUMNS[:2]):
            return existing, has_initial
        for row in reader:
            from_d = (row.get("From_Date") or "").strip()
            to_d = (row.get("To_Date") or "").strip()
            if not from_d and to_d:
                has_initial = True
            elif from_d and to_d:
                existing.add((from_d, to_d))
    return existing, has_initial


def append_new_rows_to_csv(csv_path: Path, new_rows: list[tuple[str, str, str, int, int]]) -> None:
    """
    Append new rows to existing CSV without loading full file into memory.

    Streams existing content to a temp file, appends new rows, then replaces original.
    """
    if not new_rows:
        return
    temp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as inf:
            reader = csv.reader(inf)
            with temp_path.open("w", newline="", encoding="utf-8") as outf:
                writer = csv.writer(outf)
                next(reader, None)  # skip existing header
                writer.writerow(CSV_COLUMNS)
                for row in reader:
                    writer.writerow(row)
                for r in new_rows:
                    writer.writerow(r)
        temp_path.replace(csv_path)
    except FileNotFoundError:
        with csv_path.open("w", newline="", encoding="utf-8") as outf:
            writer = csv.writer(outf)
            writer.writerow(CSV_COLUMNS)
            for r in new_rows:
                writer.writerow(r)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("signature_changes_by_district.csv"), help="Output CSV path")
    parser.add_argument("--html-dir", type=Path, default=Path("HTMLs"), help="Directory containing YYYY-MM-DD.html files")
    args = parser.parse_args()

    html_dir = args.html_dir.resolve()
    csv_path = args.csv.resolve()

    if not html_dir.is_dir():
        print(f"Error: HTML directory not found: {html_dir}", file=sys.stderr)
        return 1

    dates = get_sorted_dates(html_dir)
    if len(dates) == 0:
        print("No date files found.", file=sys.stderr)
        return 0
    
    first_date = dates[0]

    existing_pairs, has_initial_rows = read_existing_transitions(csv_path)
    consecutive_pairs = [(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]
    new_pairs = [p for p in consecutive_pairs if p not in existing_pairs]

    # Check if we need initial rows
    need_initial = not has_initial_rows
    need_transitions = len(new_pairs) > 0

    if not need_initial and not need_transitions:
        print("No new date transitions to compute. CSV is up to date.")
        return 0

    # Collect discrepancy messages
    malformed_total = 0
    duplicate_total = 0
    district_change_total = 0
    all_new_rows: list[tuple[str, str, str, int, int]] = []
    initial_rows: list[tuple[str, str, str, int, int]] = []

    # Compute initial rows if needed
    if need_initial:
        first_path = html_dir / f"{first_date}.html"
        if first_path.exists():
            first_snap, first_stats = parse_signers_table(first_path)
            malformed_total += first_stats.malformed_rows
            duplicate_total += first_stats.duplicate_ids
            if first_stats.malformed_rows:
                print(f"Skipped {first_stats.malformed_rows} malformed row(s) in {first_path.name}.", file=sys.stderr)
            if first_stats.duplicate_ids:
                print(f"Deduplicated {first_stats.duplicate_ids} duplicate Voter ID(s) in {first_path.name}.", file=sys.stderr)
            initial_rows = compute_initial_rows(first_date, first_snap)
            del first_snap

    # Dates we need to parse: any date that appears in a new transition
    dates_to_load = set()
    for a, b in new_pairs:
        dates_to_load.add(a)
        dates_to_load.add(b)

    for from_date, to_date in new_pairs:
        from_path = html_dir / f"{from_date}.html"
        to_path = html_dir / f"{to_date}.html"
        if not from_path.exists() or not to_path.exists():
            print(f"Warning: Missing file for transition {from_date} -> {to_date}", file=sys.stderr)
            continue

        prev_snap, prev_stats = parse_signers_table(from_path)
        malformed_total += prev_stats.malformed_rows
        duplicate_total += prev_stats.duplicate_ids
        if prev_stats.malformed_rows:
            print(f"Skipped {prev_stats.malformed_rows} malformed row(s) in {from_path.name}.", file=sys.stderr)
        if prev_stats.duplicate_ids:
            print(f"Deduplicated {prev_stats.duplicate_ids} duplicate Voter ID(s) in {from_path.name}.", file=sys.stderr)

        curr_snap, curr_stats = parse_signers_table(to_path)
        malformed_total += curr_stats.malformed_rows
        duplicate_total += curr_stats.duplicate_ids
        if curr_stats.malformed_rows:
            print(f"Skipped {curr_stats.malformed_rows} malformed row(s) in {to_path.name}.", file=sys.stderr)
        if curr_stats.duplicate_ids:
            print(f"Deduplicated {curr_stats.duplicate_ids} duplicate Voter ID(s) in {to_path.name}.", file=sys.stderr)

        rows, district_changes = compute_transition(from_date, to_date, prev_snap, curr_snap)
        district_change_total += district_changes
        all_new_rows.extend(rows)
        # Drop refs to free memory before next iteration
        del prev_snap, curr_snap

    if district_change_total:
        print(f"{district_change_total} Voter ID(s) had different senate district on consecutive dates.", file=sys.stderr)

    if not initial_rows and not all_new_rows:
        return 0

    # Sort rows: initial rows first (empty From_Date sorts first), then by From_Date, To_Date, Senate_District
    all_rows_to_write = initial_rows + all_new_rows
    all_rows_to_write.sort(key=lambda r: (r[0] or "0000-00-00", r[1], r[2]))

    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(all_rows_to_write)
        msg_parts = []
        if initial_rows:
            msg_parts.append(f"{len(initial_rows)} initial row(s)")
        if all_new_rows:
            msg_parts.append(f"{len(all_new_rows)} transition row(s)")
        print(f"Wrote {' and '.join(msg_parts)} to {csv_path}.")
    else:
        # If we have initial rows, we need to prepend them (they should be first)
        # Otherwise just append transitions
        if initial_rows:
            # Prepend initial rows: read existing, write header + initial + existing
            temp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
            try:
                with csv_path.open("r", newline="", encoding="utf-8") as inf:
                    reader = csv.reader(inf)
                    with temp_path.open("w", newline="", encoding="utf-8") as outf:
                        writer = csv.writer(outf)
                        next(reader, None)  # skip existing header
                        writer.writerow(CSV_COLUMNS)
                        # Write initial rows first
                        for r in initial_rows:
                            writer.writerow(r)
                        # Then existing rows
                        for row in reader:
                            writer.writerow(row)
                        # Then new transition rows
                        for r in all_new_rows:
                            writer.writerow(r)
                temp_path.replace(csv_path)
                msg_parts = []
                if initial_rows:
                    msg_parts.append(f"prepended {len(initial_rows)} initial row(s)")
                if all_new_rows:
                    msg_parts.append(f"appended {len(all_new_rows)} transition row(s)")
                print(f"{' and '.join(msg_parts)} to {csv_path}.")
            finally:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
        else:
            append_new_rows_to_csv(csv_path, all_new_rows)
            print(f"Appended {len(all_new_rows)} rows to {csv_path}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
