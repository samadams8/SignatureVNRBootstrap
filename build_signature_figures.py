#!/usr/bin/env python3
"""
Generate signature figures from signature_changes_by_district.csv.

Produces:
1. Two-panel figures per district and overall: top = signature accumulation over time,
   bottom = signatures added and removed per day (double bar chart). Saves figures/district_01.png ... district_29.png, overall.png.
2. One bar chart of removal rate by district (and overall): removal_rate_pct = 100 * sum(Removed)/sum(Added).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates


def load_and_split(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str]]]:
    """
    Load CSV and return initial rows, transition rows, and sorted list of (From_Date, To_Date) pairs.
    """
    df = pd.read_csv(csv_path, dtype={"Senate_District": str})
    df["From_Date"] = df["From_Date"].astype(str).str.strip()
    df["To_Date"] = df["To_Date"].astype(str).str.strip()
    df["Senate_District"] = df["Senate_District"].astype(str).str.strip()

    # Rows with empty or NaN From_Date are initial rows
    from_empty = df["From_Date"].isna() | (df["From_Date"].astype(str).str.strip() == "")
    initial = df[from_empty].copy()
    transitions = df[~from_empty].copy()

    # Unique transition pairs sorted by To_Date
    pairs = transitions[["From_Date", "To_Date"]].drop_duplicates()
    pairs = pairs.sort_values("To_Date").apply(tuple, axis=1).tolist()

    return initial, transitions, pairs


def build_series(
    initial: pd.DataFrame,
    transitions: pd.DataFrame,
    ordered_pairs: list[tuple[str, str]],
    district: str | None,
) -> tuple[list[pd.Timestamp], list[float], list[pd.Timestamp], list[int], list[int]]:
    """
    Build accumulation and daily added/removed series for one district or overall.

    Returns (acc_dates, acc_cumulative, daily_dates, added_values, removed_values).
    If district is None, aggregate over all districts (overall).
    """
    if district is not None:
        init = initial[initial["Senate_District"] == district]
        trans = transitions[transitions["Senate_District"] == district]
    else:
        init = initial.groupby("To_Date", as_index=False).agg({"Added": "sum", "Removed": "sum"})
        trans = transitions.groupby(["From_Date", "To_Date"], as_index=False).agg({"Added": "sum", "Removed": "sum"})

    if init.empty:
        return [], [], [], [], []

    first_date = init["To_Date"].iloc[0]
    cum = int(init["Added"].sum() - init["Removed"].sum())
    
    acc_dates = [pd.Timestamp(first_date)]
    acc_cum = [cum]
    
    daily_dates = []
    added_values = []
    removed_values = []

    for from_d, to_d in ordered_pairs:
        if district is not None:
            row = trans[(trans["From_Date"] == from_d) & (trans["To_Date"] == to_d)]
        else:
            row = trans[(trans["From_Date"] == from_d) & (trans["To_Date"] == to_d)]
        if row.empty:
            continue
        added = int(row["Added"].iloc[0])
        removed = int(row["Removed"].iloc[0])
        net = added - removed
        cum += net
        acc_dates.append(pd.Timestamp(to_d))
        acc_cum.append(cum)
        daily_dates.append(pd.Timestamp(to_d))
        added_values.append(added)
        removed_values.append(removed)

    return acc_dates, acc_cum, daily_dates, added_values, removed_values


def plot_two_panel(
    acc_dates: list[pd.Timestamp],
    acc_cum: list[float],
    daily_dates: list[pd.Timestamp],
    added_values: list[int],
    removed_values: list[int],
    title: str,
    out_path: Path,
    fmt: str = "png",
) -> None:
    """Create a two-panel figure: top = accumulation over time, bottom = added/removed per day (double bar chart)."""
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=False, figsize=(10, 6))

    # Top: accumulation over time
    ax_top.plot(acc_dates, acc_cum, marker="o", markersize=3)
    ax_top.set_ylabel("Cumulative signatures")
    ax_top.set_title("Signature accumulation over time")
    ax_top.grid(True, alpha=0.3)
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_top.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax_top.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Bottom: double bar chart showing additions and removals per day
    x_pos = range(len(daily_dates))
    width = 0.35
    
    ax_bot.bar([x - width/2 for x in x_pos], added_values, width, label="Added", alpha=0.7, color="steelblue")
    ax_bot.bar([x + width/2 for x in x_pos], removed_values, width, label="Removed", alpha=0.7, color="coral")
    ax_bot.set_ylabel("Signatures")
    ax_bot.set_xlabel("Date")
    ax_bot.set_title("Signatures added and removed per day")
    ax_bot.set_xticks(x_pos)
    ax_bot.set_xticklabels([d.strftime("%Y-%m-%d") for d in daily_dates], rotation=45, ha="right")
    ax_bot.legend()
    ax_bot.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, format=fmt, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_removal_rate(
    districts: list[str],
    rates: list[float],
    out_path: Path,
    fmt: str = "png",
) -> None:
    """Bar chart of removal rate (%) by district (and Overall)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(districts))
    bars = ax.bar(x, rates, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(districts, rotation=45, ha="right")
    ax.set_ylabel("Removal rate (%)")
    ax.set_xlabel("Senate district")
    ax.set_title("Signature removal rate (total removed / total collected)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, format=fmt, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("signature_changes_by_district.csv"), help="Input CSV path")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"), help="Output directory for figures")
    parser.add_argument("--format", choices=["png", "pdf"], default="png", help="Output format (default: png)")
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    out_dir = args.out_dir.resolve()
    fmt = args.format

    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    initial, transitions, ordered_pairs = load_and_split(csv_path)
    if not ordered_pairs:
        print("No transition data found.", file=sys.stderr)
        return 0

    # District list: 1–29 as strings, sorted numerically
    district_list = sorted(
        set(initial["Senate_District"].dropna().unique()) | set(transitions["Senate_District"].dropna().unique()),
        key=int,
    )

    # Two-panel figure per district
    for d in district_list:
        acc_dates, acc_cum, daily_dates, added_values, removed_values = build_series(initial, transitions, ordered_pairs, d)
        if not acc_dates:
            continue
        out_path = out_dir / f"district_{int(d):02d}.{fmt}"
        plot_two_panel(acc_dates, acc_cum, daily_dates, added_values, removed_values, f"District {d}", out_path, fmt)

    # Overall two-panel
    acc_dates, acc_cum, daily_dates, added_values, removed_values = build_series(initial, transitions, ordered_pairs, None)
    if acc_dates:
        out_path = out_dir / f"overall.{fmt}"
        plot_two_panel(acc_dates, acc_cum, daily_dates, added_values, removed_values, "Overall", out_path, fmt)

    # Removal rate by district
    removal_rates = []
    labels = []
    for d in district_list:
        mask_init = (initial["Senate_District"] == d)
        mask_trans = (transitions["Senate_District"] == d)
        total_added = initial.loc[mask_init, "Added"].sum() + transitions.loc[mask_trans, "Added"].sum()
        total_removed = initial.loc[mask_init, "Removed"].sum() + transitions.loc[mask_trans, "Removed"].sum()
        if total_added == 0:
            rate = 0.0
        else:
            rate = 100.0 * total_removed / total_added
        removal_rates.append(rate)
        labels.append(d)

    # Overall removal rate
    total_added = initial["Added"].sum() + transitions["Added"].sum()
    total_removed = initial["Removed"].sum() + transitions["Removed"].sum()
    removal_rates.append(100.0 * total_removed / total_added if total_added else 0.0)
    labels.append("Overall")

    plot_removal_rate(labels, removal_rates, out_dir / f"removal_rate_by_district.{fmt}", fmt)

    print(f"Figures written to {out_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
