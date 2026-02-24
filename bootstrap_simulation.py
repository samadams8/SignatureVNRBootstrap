#!/usr/bin/env python3
"""
Monte Carlo bootstrap simulation for signature processing.

Projects final signature counts per district accounting for:
- Future signature verification through gathering deadline
- Signature removals through removal deadline (45-day window)
- Two sampling modes: naive (all historical data) and filtered (filtered by threshold)
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# District targets (hardcoded per specification)
DISTRICT_TARGETS = {
    1: 5238, 2: 4687, 3: 4737, 4: 5099, 5: 4115, 
    6: 4745, 7: 5294, 8: 4910, 9: 4805, 10: 2975, 
    11: 4890, 12: 3248, 13: 4088, 14: 5680, 15: 4596, 
    16: 4347, 17: 5368, 18: 5093, 19: 5715, 20: 5292, 
    21: 5684, 22: 5411, 23: 4253, 24: 3857, 25: 4929, 
    26: 5178, 27: 5696, 28: 5437, 29: 5382
}
STATEWIDE_TARGET = 140748
DISTRICTS_TARGET = 26

class SimulationData(NamedTuple):
    """Data structures for simulation."""
    
    naive_counts: dict[str, int]  # District -> initial count
    added_pools: dict[str, list[int]]  # District -> list of Added values
    removed_pools: dict[str, list[int]]  # District -> list of Removed values
    start_date: date  # Latest To_Date in CSV
    filtered_pools: dict[str, list[int]] | None  # District -> filtered Added values

class SimulationResults(NamedTuple):
    """Results from a single simulation run."""
    
    final_counts: dict[str, int]  # District -> final count
    total_statewide: int
    districts_meeting_target: int

class AggregatedResults(NamedTuple):
    """Aggregated results across all trials."""
    
    statewide_totals: list[int]  # Total signatures across all trials
    districts_meeting: list[int]  # Number of districts meeting targets per trial
    district_success_counts: dict[str, int]  # District -> number of trials meeting target

def load_data(csv_path: Path, removal_cutoff_date: date | None = None) -> SimulationData:
    """
    Load CSV and prepare data structures for simulation.
    
    Parameters
    ----------
    csv_path : Path
        Path to signature_changes_by_district.csv
    removal_cutoff_date : date | None
        Only include removals from dates after this cutoff (None = include all)
        
    Returns
    -------
    SimulationData
        Data structures containing naive counts, pools, and start date
    """
    df = pd.read_csv(csv_path, dtype={"Senate_District": str})
    df["From_Date"] = df["From_Date"].astype(str).str.strip()
    df["To_Date"] = df["To_Date"].astype(str).str.strip()
    df["Senate_District"] = df["Senate_District"].astype(str).str.strip()
    
    # Separate initial rows (empty From_Date) from transition rows
    from_empty = df["From_Date"].isna() | (df["From_Date"].astype(str).str.strip() == "")
    initial_df = df[from_empty].copy()
    transitions_df = df[~from_empty].copy()
    
    # Calculate naive counts per district from initial rows
    naive_counts: dict[str, int] = {}
    for _, row in initial_df.iterrows():
        district = row["Senate_District"]
        added = int(row["Added"])
        removed = int(row["Removed"])
        naive_counts[district] = naive_counts.get(district, 0) + added - removed
    
    # Build sampling pools from transition rows AND add transitions to naive counts
    # Naive counts should represent CURRENT state (initial + all transitions)
    added_pools: dict[str, list[int]] = defaultdict(list)
    removed_pools: dict[str, list[int]] = defaultdict(list)
    
    # Parse To_Date for filtering removals
    transitions_df["To_Date_parsed"] = pd.to_datetime(transitions_df["To_Date"], errors="coerce")
    
    for _, row in transitions_df.iterrows():
        district = row["Senate_District"]
        added = int(row["Added"])
        removed = int(row["Removed"])
        to_date = row["To_Date_parsed"]
        
        # Add to naive counts (current state)
        naive_counts[district] = naive_counts.get(district, 0) + added - removed
        
        # Add to sampling pools (for future simulation)
        added_pools[district].append(added)
        
        # Only include removals from dates on or after cutoff (if specified)
        if removed > 0:
            if removal_cutoff_date is None:
                # Include all removals
                removed_pools[district].append(removed)
            else:
                # Only include if To_Date is on or after cutoff (inclusive)
                if not pd.isna(to_date) and to_date.date() >= removal_cutoff_date:
                    removed_pools[district].append(removed)
    
    # Convert defaultdicts to regular dicts
    added_pools = dict(added_pools)
    removed_pools = dict(removed_pools)
    
    # Find latest To_Date as simulation start date
    all_dates = pd.to_datetime(transitions_df["To_Date"], errors="coerce")
    latest_date = all_dates.max()
    if pd.isna(latest_date):
        # Fallback to initial rows if no transitions
        initial_dates = pd.to_datetime(initial_df["To_Date"], errors="coerce")
        latest_date = initial_dates.max()
    
    start_date = latest_date.date() if not pd.isna(latest_date) else date.today()
    
    return SimulationData(
        naive_counts=naive_counts,
        added_pools=added_pools,
        removed_pools=removed_pools,
        start_date=start_date,
        filtered_pools=None
    )

def calculate_filter_thresholds(
    csv_path: Path,
    quantile: float = 0.5,
    window_days: int = 7
) -> dict[str, float]:
    """
    Calculate filter threshold for each district.
    
    Parameters
    ----------
    csv_path : Path
        Path to CSV file
    quantile : float
        Quantile to use for threshold (default: 0.5 for median)
    window_days : int
        Number of days to look back (default: 7)
        
    Returns
    -------
    dict[str, float]
        District -> threshold value
    """
    df = pd.read_csv(csv_path, dtype={"Senate_District": str})
    df["From_Date"] = df["From_Date"].astype(str).str.strip()
    df["To_Date"] = df["To_Date"].astype(str).str.strip()
    df["Senate_District"] = df["Senate_District"].astype(str).str.strip()
    
    # Only use transition rows (exclude initial rows)
    from_empty = df["From_Date"].isna() | (df["From_Date"].astype(str).str.strip() == "")
    transitions_df = df[~from_empty].copy()
    
    transitions_df["To_Date"] = pd.to_datetime(transitions_df["To_Date"])
    transitions_df = transitions_df.sort_values("To_Date")
    
    thresholds: dict[str, float] = {}
    
    # Get unique districts
    districts = transitions_df["Senate_District"].unique()
    
    for district in districts:
        district_data = transitions_df[transitions_df["Senate_District"] == district].copy()
        
        if len(district_data) == 0:
            thresholds[district] = 0.0
            continue
        
        # Get last window_days days with recorded data
        unique_dates = district_data["To_Date"].unique()
        unique_dates = sorted(unique_dates)
        
        if len(unique_dates) == 0:
            thresholds[district] = 0.0
            continue
        
        # Take last window_days unique dates
        last_dates = unique_dates[-window_days:] if len(unique_dates) >= window_days else unique_dates
        
        # Extract Added values from those dates
        recent_added = district_data[district_data["To_Date"].isin(last_dates)]["Added"].tolist()
        
        if len(recent_added) == 0:
            thresholds[district] = 0.0
        else:
            threshold = np.quantile(recent_added, quantile)
            thresholds[district] = float(threshold)
    
    return thresholds

def build_filtered_pools(
    added_pools: dict[str, list[int]],
    thresholds: dict[str, float]
) -> dict[str, list[int]]:
    """
    Build filtered sampling pools by filtering Added values >= threshold.
    
    Parameters
    ----------
    added_pools : dict[str, list[int]]
        Naive Added pools per district
    thresholds : dict[str, float]
        Filter thresholds per district
        
    Returns
    -------
    dict[str, list[int]]
        Filtered Added pools per district
    """
    filtered_pools: dict[str, list[int]] = {}
    
    for district, pool in added_pools.items():
        threshold = thresholds.get(district, 0.0)
        # Filter: keep values >= threshold (0s excluded if below threshold)
        filtered = [v for v in pool if v >= threshold]
        # If filtered pool is empty, use original pool to avoid sampling errors
        filtered_pools[district] = filtered if filtered else pool
    
    return filtered_pools


def run_single_trial(
    data: SimulationData,
    gathering_deadline: date,
    removal_deadline: date,
    removal_window_days: int,
    use_filtered: bool = False,
    disable_removals: bool = False
) -> SimulationResults:
    """
    Run a single Monte Carlo trial.
    
    Parameters
    ----------
    data : SimulationData
        Preprocessed data structures
    gathering_deadline : date
        Last day for signature verification
    removal_deadline : date
        Last day for signature removal
    removal_window_days : int
        Number of days signatures are eligible for removal (45)
    use_filtered : bool
        If True, use filtered pools; otherwise use naive pools
    disable_removals : bool
        If True, skip removal sampling phase (keep historical removals in naive counts)
        
    Returns
    -------
    SimulationResults
        Final counts, statewide total, and districts meeting targets
    """
    # Initialize running totals from naive counts
    # Include all districts that appear in either naive counts or pools
    all_districts = set(data.naive_counts.keys()) | set(data.added_pools.keys())
    current_counts: dict[str, int] = {
        district: data.naive_counts.get(district, 0)
        for district in all_districts
    }
    
    # Choose which pools to use
    added_pools = data.filtered_pools if (use_filtered and data.filtered_pools) else data.added_pools
    
    # Track verification dates for removable pool calculation
    # Structure: district -> list of (date, count) tuples
    verification_history: dict[str, list[tuple[date, int]]] = defaultdict(list)
    
    # Phase 1: Signature Verification (start_date through gathering_deadline)
    current_date = data.start_date
    while current_date <= gathering_deadline:
        for district in all_districts:
            # Get sampling pool for this district
            pool = added_pools.get(district, [0])
            
            # Sample one Added value
            sampled_added = int(np.random.choice(pool))
            
            # Add to running total
            current_counts[district] += sampled_added
            
            # Track verification date for removable pool
            if sampled_added > 0:
                verification_history[district].append((current_date, sampled_added))
        
        current_date += timedelta(days=1)
    
    # Phase 2: Signature Removals (start_date through removal_deadline)
    # Skip this phase if removals are disabled
    if not disable_removals:
        current_date = data.start_date
        while current_date <= removal_deadline:
            for district in all_districts:
                # Calculate removable pool: signatures verified within last removal_window_days (including same day)
                # If removal_window_days=45, we want days (current_date - 44) through current_date (inclusive)
                cutoff_date = current_date - timedelta(days=removal_window_days - 1)
                removable_pool = 0
                
                for verify_date, count in verification_history[district]:
                    if verify_date >= cutoff_date:
                        removable_pool += count
                
                # Cap removable pool at current total (can't remove more than exists)
                removable_pool = min(removable_pool, current_counts[district])
                
                # Sample one Removed value from historical distribution
                removed_pool = data.removed_pools.get(district, [0])
                sampled_removed = int(np.random.choice(removed_pool))
                
                # Cap sampled removal by removable pool
                actual_removed = min(sampled_removed, removable_pool)
                
                # Subtract from running total
                current_counts[district] -= actual_removed
            
            current_date += timedelta(days=1)
    
    # Calculate results
    total_statewide = sum(current_counts.values())
    
    # Count districts meeting their targets
    districts_meeting = 0
    for district, count in current_counts.items():
        target = DISTRICT_TARGETS.get(int(district), 0)
        if count >= target:
            districts_meeting += 1
    
    return SimulationResults(
        final_counts=current_counts,
        total_statewide=total_statewide,
        districts_meeting_target=districts_meeting
    )

def run_simulation(
    data: SimulationData,
    n_trials: int,
    gathering_deadline: date,
    removal_deadline: date,
    removal_window_days: int,
    use_filtered: bool = False,
    disable_removals: bool = False,
    seed: int | None = None
) -> AggregatedResults:
    """
    Run full Monte Carlo simulation.
    
    Parameters
    ----------
    data : SimulationData
        Preprocessed data structures
    n_trials : int
        Number of Monte Carlo trials
    gathering_deadline : date
        Last day for signature verification
    removal_deadline : date
        Last day for signature removal
    removal_window_days : int
        Number of days signatures are eligible for removal
    use_filtered : bool
        If True, use filtered pools
    disable_removals : bool
        If True, skip removal sampling phase
    seed : int | None
        Random seed for reproducibility
        
    Returns
    -------
    AggregatedResults
        Statewide totals, districts meeting targets, and per-district success counts
    """
    if seed is not None:
        np.random.seed(seed)

    all_districts = sorted(
        set(data.naive_counts.keys()) | set(data.added_pools.keys()),
        key=lambda d: int(d)
    )

    if n_trials <= 0 or len(all_districts) == 0:
        return AggregatedResults(
            statewide_totals=[],
            districts_meeting=[],
            district_success_counts={district: 0 for district in all_districts}
        )

    n_districts = len(all_districts)
    n_verification_days = (gathering_deadline - data.start_date).days + 1
    n_verification_days = max(0, n_verification_days)
    n_removal_days = (removal_deadline - data.start_date).days + 1
    n_removal_days = max(0, n_removal_days)

    initial_counts = np.array(
        [data.naive_counts.get(district, 0) for district in all_districts],
        dtype=np.int64
    )
    final_counts_matrix = np.tile(initial_counts, (n_trials, 1))

    added_pools = data.filtered_pools if (use_filtered and data.filtered_pools) else data.added_pools

    for district_idx, district in enumerate(all_districts):
        district_added_pool = np.array(added_pools.get(district, [0]), dtype=np.int64)

        if n_verification_days > 0:
            sampled_added = np.random.choice(
                district_added_pool,
                size=(n_trials, n_verification_days)
            )
            final_counts_matrix[:, district_idx] += sampled_added.sum(axis=1)
        else:
            sampled_added = np.zeros((n_trials, 0), dtype=np.int64)

        if disable_removals or n_removal_days == 0:
            continue

        district_removed_pool = np.array(data.removed_pools.get(district, [0]), dtype=np.int64)
        sampled_removed = np.random.choice(
            district_removed_pool,
            size=(n_trials, n_removal_days)
        )

        if n_verification_days == 0:
            removable_pool_by_day = np.zeros((n_trials, n_removal_days), dtype=np.int64)
        else:
            # Reproduce existing removal-pool logic:
            # removable_pool(day t) = sum of all verification adds with verify_date >= cutoff_date(t)
            suffix_sums = np.cumsum(sampled_added[:, ::-1], axis=1)[:, ::-1]
            suffix_with_zero = np.concatenate(
                [suffix_sums, np.zeros((n_trials, 1), dtype=np.int64)],
                axis=1
            )

            cutoff_indices = np.arange(n_removal_days) - (removal_window_days - 1)
            clipped_cutoff_indices = np.clip(cutoff_indices, 0, n_verification_days)
            removable_pool_by_day = suffix_with_zero[:, clipped_cutoff_indices]

        district_counts = final_counts_matrix[:, district_idx].copy()
        for day_idx in range(n_removal_days):
            capped_removable = np.minimum(removable_pool_by_day[:, day_idx], district_counts)
            actual_removed = np.minimum(sampled_removed[:, day_idx], capped_removable)
            district_counts -= actual_removed

        final_counts_matrix[:, district_idx] = district_counts

    statewide_totals_arr = final_counts_matrix.sum(axis=1)

    district_targets = np.array(
        [DISTRICT_TARGETS.get(int(district), 0) for district in all_districts],
        dtype=np.int64
    )
    success_matrix = final_counts_matrix >= district_targets

    districts_meeting_arr = success_matrix.sum(axis=1)
    district_success_counts = {
        district: int(success_matrix[:, district_idx].sum())
        for district_idx, district in enumerate(all_districts)
    }

    return AggregatedResults(
        statewide_totals=statewide_totals_arr.astype(int).tolist(),
        districts_meeting=districts_meeting_arr.astype(int).tolist(),
        district_success_counts=district_success_counts
    )

def plot_histogram(
    values: list[int],
    threshold: int,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    fmt: str = "png",
    bins: int | list[int] | None = None,
    xlim: tuple[int, int] | None = None
) -> None:
    """
    Generate histogram with reference line.
    
    Parameters
    ----------
    values : list[int]
        Values to plot
    threshold : int
        Reference threshold line
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    out_path : Path
        Output file path
    fmt : str
        Output format (png or pdf)
    bins : int | list[int] | None
        Number of bins or explicit bin edges. If None, uses 50 bins.
    xlim : tuple[int, int] | None
        X-axis limits (min, max). If None, auto-scales.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram with relative frequency (%)
    if bins is None:
        bins = 50
    
    # Calculate weights to convert counts to percentages
    total_count = len(values)
    weights = [100.0 / total_count] * total_count if total_count > 0 else []
    
    ax.hist(values, bins=bins, weights=weights, alpha=0.7, edgecolor="black")
    
    # Set x-axis limits if specified
    if xlim is not None:
        ax.set_xlim(xlim)
    
    # Add reference line
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Target: {threshold:,}")
    
    # Add summary statistics
    mean_val = np.mean(values)
    median_val = np.median(values)
    p5 = np.percentile(values, 5)
    p95 = np.percentile(values, 95)
    
    stats_text = f"Mean: {mean_val:,.0f}\nMedian: {median_val:,.0f}\n5th: {p5:,.0f}\n95th: {p95:,.0f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, format=fmt, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_district_success_probability(
    district_success_counts: dict[str, int],
    n_trials: int,
    out_path: Path,
    fmt: str = "png"
) -> None:
    """
    Generate bar chart showing probability of success for each district.
    
    Parameters
    ----------
    district_success_counts : dict[str, int]
        District -> count of successful trials
    n_trials : int
        Total number of trials
    out_path : Path
        Output file path
    fmt : str
        Output format (png or pdf)
    """
    # Convert to sorted district numbers and their success rates
    districts = sorted([int(d) for d in district_success_counts.keys()])
    probabilities = [
        100.0 * district_success_counts[str(d)] / n_trials
        for d in districts
    ]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create bar chart with color gradient based on probability
    colors = []
    for prob in probabilities:
        if prob >= 95:
            colors.append("darkgreen")
        elif prob >= 80:
            colors.append("green")
        elif prob >= 50:
            colors.append("gold")
        elif prob >= 20:
            colors.append("orange")
        else:
            colors.append("red")
    
    bars = ax.bar(districts, probabilities, color=colors, edgecolor="black", linewidth=1.5)
    
    # Add value labels on top of bars
    for i, (district, prob) in enumerate(zip(districts, probabilities)):
        ax.text(district, prob + 1.5, f"{prob:.0f}%", ha="center", va="bottom", fontsize=9)
    
    # Add reference line at 50% and 80%
    ax.axhline(50, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="50% threshold")
    ax.axhline(80, color="green", linestyle="--", linewidth=1, alpha=0.7, label="80% threshold")
    
    ax.set_xlabel("Senate District", fontsize=12, fontweight="bold")
    ax.set_ylabel("Probability of Meeting Target (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"District Success Probability (based on {n_trials:,} trials)", 
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_xticks(districts)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    
    fig.tight_layout()
    fig.savefig(out_path, format=fmt, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_parameter_summary(
    csv_path: Path,
    removal_cutoff_date: date | None,
    out_path: Path,
    fmt: str = "png",
    filter_quantile: float | None = None,
    filter_window: int | None = None
) -> None:
    """
    Generate parameter summary figure showing daily add/removal rates and thresholds.
    
    This is a diagnostic figure and does not affect simulation logic.
    The threshold is calculated from overall daily totals for visualization purposes only.
    
    Parameters
    ----------
    csv_path : Path
        Path to CSV file
    removal_cutoff_date : date | None
        Removal cutoff date (vertical line)
    out_path : Path
        Output file path
    fmt : str
        Output format (png or pdf)
    filter_quantile : float | None
        Filter quantile used (for calculating diagnostic threshold from overall daily totals)
    filter_window : int | None
        Filter window days used (for calculating diagnostic threshold from overall daily totals)
    """
    df = pd.read_csv(csv_path, dtype={"Senate_District": str})
    df["From_Date"] = df["From_Date"].astype(str).str.strip()
    df["To_Date"] = df["To_Date"].astype(str).str.strip()
    
    # Only use transition rows (exclude initial rows)
    from_empty = df["From_Date"].isna() | (df["From_Date"].astype(str).str.strip() == "")
    transitions_df = df[~from_empty].copy()
    
    # Parse dates
    transitions_df["To_Date"] = pd.to_datetime(transitions_df["To_Date"], errors="coerce")
    transitions_df = transitions_df.sort_values("To_Date")
    
    # Aggregate by date (sum across all districts)
    daily_totals = transitions_df.groupby("To_Date").agg({
        "Added": "sum",
        "Removed": "sum"
    }).reset_index()
    
    # Calculate diagnostic filter threshold from overall daily totals (if filtered mode)
    filter_threshold = None
    if filter_quantile is not None and filter_window is not None:
        if len(daily_totals) >= filter_window:
            last_days = daily_totals.tail(filter_window)["Added"].values
            filter_threshold = float(np.quantile(last_days, filter_quantile))
    
    # Create figure with two panels (upper: additions, lower: removals)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Upper panel: Daily Added
    ax1.plot(daily_totals["To_Date"], daily_totals["Added"], 
             label="Daily Added", color="blue", linewidth=2, marker="o", markersize=3)
    ax1.set_ylabel("Daily Added Signatures", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add horizontal line for filter threshold if provided
    if filter_threshold is not None:
        ax1.axhline(filter_threshold, color="green", linestyle="--", linewidth=2, label="Filter Threshold (inclusive)")
    
    ax1.legend(loc="upper left")
    ax1.set_title("Parameter Summary: Daily Add/Removal Rates and Thresholds", fontsize=14, fontweight="bold")
    
    # Lower panel: Daily Removed
    ax2.plot(daily_totals["To_Date"], daily_totals["Removed"],
             label="Daily Removed", color="orange", linewidth=2, marker="s", markersize=3)
    ax2.set_ylabel("Daily Removed Signatures", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add vertical line for removal cutoff date (also on lower panel)
    if removal_cutoff_date is not None:
        cutoff_timestamp = pd.Timestamp(removal_cutoff_date)
        ax2.axvline(cutoff_timestamp, color="red", linestyle="--", linewidth=2,
                   label=f"Inclusion cutoff")
    
    ax2.legend(loc="upper left")
    
    # Format x-axis dates (only on lower panel since sharex=True)
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    fig.tight_layout()
    fig.savefig(out_path, format=fmt, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("signature_changes_by_district.csv"),
        help="Path to CSV file"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100000,
        help="Number of Monte Carlo trials"
    )
    parser.add_argument(
        "--mode",
        choices=["naive", "filtered"],
        default="filtered",
        help="Sampling mode (default: filtered)"
    )
    parser.add_argument(
        "--filter-quantile",
        type=float,
        default=0.5,
        help="Quantile for filter threshold"
    )
    parser.add_argument(
        "--filter-window",
        type=int,
        default=14,
        help="Days for filter threshold calculation"
    )
    parser.add_argument(
        "--gathering-deadline",
        type=str,
        default="2026-03-08",
        help="Gathering deadline date (default: 2026-03-08)"
    )
    parser.add_argument(
        "--removal-deadline",
        type=str,
        default="2026-04-22",
        help="Removal deadline date"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bootstrap_output"),
        help="Output directory for figures"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf"],
        default="png",
        help="Output format (default: png)"
    )
    parser.add_argument(
        "--no-removals",
        action="store_true",
        help="Disable removal sampling (keep historical removals in naive counts only)"
    )
    parser.add_argument(
        "--removal-cutoff-window",
        type=int,
        default=14,
        help="Number of days to look back for removal sampling cutoff"
    )
    parser.add_argument(
        "--removal-cutoff-date",
        type=str,
        default=None,
        help="Only sample removals from dates after this cutoff; overrides --removal-cutoff-window if specified (format: YYYY-MM-DD)"
    )
    
    args = parser.parse_args()

    # If removal cutoff date is not provided, calculate it from the window
    if args.removal_cutoff_date is None:
        args.removal_cutoff_date = (date.today() - timedelta(days=args.removal_cutoff_window)).isoformat()
    
    # Validate inputs
    csv_path = args.csv.resolve()
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1
    
    try:
        gathering_deadline = date.fromisoformat(args.gathering_deadline)
        removal_deadline = date.fromisoformat(args.removal_deadline)
        removal_cutoff_date = date.fromisoformat(args.removal_cutoff_date)
    except ValueError as e:
        print(f"Error: Invalid date format: {e}", file=sys.stderr)
        return 1
    
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...", file=sys.stderr)
    data = load_data(csv_path, removal_cutoff_date=removal_cutoff_date)
    print(f"Loaded data. Start date: {data.start_date}", file=sys.stderr)
    if removal_cutoff_date:
        print(f"Removal sampling cutoff: only using removals on or after {removal_cutoff_date}", file=sys.stderr)
    
    # Calculate filter thresholds if needed
    use_filtered = (args.mode == "filtered")
    if use_filtered:
        print("Calculating filter thresholds...", file=sys.stderr)
        thresholds = calculate_filter_thresholds(
            csv_path,
            quantile=args.filter_quantile,
            window_days=args.filter_window
        )
        filtered_pools = build_filtered_pools(data.added_pools, thresholds)
        data = data._replace(filtered_pools=filtered_pools)
        print("Filter thresholds calculated.", file=sys.stderr)
    
    # Generate descriptive mode label (quantile as percentage: 0.5 -> 50)
    if use_filtered:
        quantile_pct = int(args.filter_quantile * 100)
        mode_label = f"filtered_q{quantile_pct}_w{args.filter_window}"
    else:
        mode_label = "naive"
    
    # Run simulation
    removal_status = " (removals disabled)" if args.no_removals else ""
    print(f"\nRunning {mode_label} simulation ({args.trials} trials){removal_status}...", file=sys.stderr)
    aggregated_results = run_simulation(
        data,
        args.trials,
        gathering_deadline,
        removal_deadline,
        removal_window_days=45,
        use_filtered=use_filtered,
        disable_removals=args.no_removals,
        seed=args.seed
    )
    
    statewide_totals = aggregated_results.statewide_totals
    districts_meeting = aggregated_results.districts_meeting
    district_success_counts = aggregated_results.district_success_counts
    
    # Generate parameter summary figure (diagnostic only, doesn't affect simulation)
    print("Generating parameter summary figure...", file=sys.stderr)
    param_summary_path = output_dir / f"{mode_label}_parameter_summary.{args.format}"
    plot_parameter_summary(
        csv_path,
        removal_cutoff_date=removal_cutoff_date,
        out_path=param_summary_path,
        fmt=args.format,
        filter_quantile=args.filter_quantile if use_filtered else None,
        filter_window=args.filter_window if use_filtered else None
    )
    
    # Generate district success probability figure
    print(f"Generating district success probability figure...", file=sys.stderr)
    district_prob_path = output_dir / f"{mode_label}_district_success_probability.{args.format}"
    plot_district_success_probability(
        district_success_counts,
        args.trials,
        district_prob_path,
        args.format
    )
    
    # Generate histograms
    print(f"Generating histograms for {mode_label} mode...", file=sys.stderr)
    
    # Create descriptive title suffix (quantile as percentage)
    if use_filtered:
        title_suffix = f"Filtered q={args.filter_quantile:.0%}, w={args.filter_window}"
    else:
        title_suffix = "Naive"
    
    # Histogram 1: Statewide totals
    statewide_path = output_dir / f"{mode_label}_statewide_histogram.{args.format}"
    # Calculate bins that are always 5000 wide
    min_val = min(statewide_totals) if statewide_totals else 0
    max_val = max(statewide_totals) if statewide_totals else 0
    BIN_WIDTH = 1000
    # Round down min to nearest BIN_WIDTH, round up max to nearest BIN_WIDTH
    bin_start = (min_val // BIN_WIDTH) * BIN_WIDTH
    bin_end = ((max_val + BIN_WIDTH - 1) // BIN_WIDTH) * BIN_WIDTH  # Round up to nearest BIN_WIDTH
    # Create bin edges at BIN_WIDTH intervals
    statewide_bins = list(range(bin_start, bin_end + BIN_WIDTH, BIN_WIDTH))
    plot_histogram(
        statewide_totals,
        STATEWIDE_TARGET,
        f"Distribution of Total Signatures Statewide ({title_suffix})",
        "Total Signatures Statewide",
        "Relative Frequency (%)",
        statewide_path,
        args.format,
        bins=statewide_bins
    )
    
    # Histogram 2: Districts meeting targets
    districts_path = output_dir / f"{mode_label}_districts_histogram.{args.format}"
    # Use bins centered on integer values 0 through 29
    # Bin edges: [-0.5, 0.5), [0.5, 1.5), ..., [28.5, 29.5)
    plot_histogram(
        districts_meeting,
        DISTRICTS_TARGET,
        f"Distribution of Districts Meeting Individual Targets ({title_suffix})",
        "Number of Districts Meeting Targets",
        "Relative Frequency (%)",
        districts_path,
        args.format,
        bins=[x - 0.5 for x in range(0, 31)],  # Bins centered on 0, 1, 2, ..., 29 (edges: -0.5 to 29.5)
        xlim=(-0.5, 29.5)  # Show range centered on 0-29
    )
    
    print(f"All figures saved to {output_dir}", file=sys.stderr)
    
    print("\nSimulation complete!", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
