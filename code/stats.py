import argparse
import csv
import sys
import colorsys
import numpy as np
from math import atan2, sqrt, pi
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import matplotlib
import re
from collections import Counter

def read_rgb(csv_path):
    rgb = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                # assume header; if not, the first line will be skipped
                continue
            if len(row) < 3:
                continue
            try:
                r, g, b = int(row[0]), int(row[1]), int(row[2])
            except ValueError:
                # skip rows that can't be parsed as ints
                continue
            rgb.append([r, g, b])
    if len(rgb) == 0:
        raise ValueError("No RGB rows found in CSV")
    return np.array(rgb, dtype=int)

def circular_stats_deg(angles_deg):
    """Return circular mean (deg), R (mean resultant length), circular variance, circular std (deg)."""
    angles_rad = np.deg2rad(angles_deg)
    sin_sum = np.sum(np.sin(angles_rad))
    cos_sum = np.sum(np.cos(angles_rad))
    n = angles_rad.size
    mean_angle_rad = atan2(sin_sum, cos_sum)
    mean_angle_deg = np.rad2deg(mean_angle_rad) % 360
    R = sqrt(sin_sum**2 + cos_sum**2) / n
    circ_var = 1 - R
    if R > 0:
        circ_std_rad = sqrt(-2 * np.log(R))
        circ_std_deg = np.rad2deg(circ_std_rad)
    else:
        circ_std_deg = float("nan")
    return mean_angle_deg, R, circ_var, circ_std_deg

def list_unique_color_terms(csv_path, sort_alpha=False):
    """
    Print unique color terms and frequencies for the requested columns.
    Looks up columns case-insensitively in the CSV header.
    If sort_alpha is True, terms are listed alphabetically; otherwise by descending frequency.
    """
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        cols = reader.fieldnames[4:]
        counters = {col: Counter() for col in cols}
        # map lower-case header -> actual header to allow case-insensitive matches
        header_map = {name.lower(): name for name in reader.fieldnames}
        actual_cols = {}
        missing = []
        for col in cols:
            if col in header_map:
                actual_cols[col] = header_map[col]
            else:
                # try literal match as fallback
                if col in reader.fieldnames:
                    actual_cols[col] = col
                else:
                    missing.append(col)
        if missing:
            print(f"Warning: missing columns in CSV header: {missing}", file=sys.stderr)
        for row in reader:
            for col in cols:
                if col not in actual_cols:
                    continue
                raw = row.get(actual_cols[col], "")
                key = raw.lower()
                counters[col][key] += 1

    # Print results
    for col in cols:
        cnt = counters[col]
        total_cells = sum(cnt.values())
        distinct = len(cnt)
        print(f"\nUnique terms in column '{col}': total terms found = {total_cells}, distinct = {distinct}")
        if distinct == 0:
            print("  (none)")
            continue
        if sort_alpha:
            for term, freq in sorted(cnt.items(), key=lambda x: x[0]):
                print(f"  {term}  ({freq})")
        else:
            for term, freq in cnt.most_common():
                print(f"  {term}  ({freq})")

def main():
    p = argparse.ArgumentParser(description="Compute statistics on RGB data; optional hue (circular) analysis.")
    p.add_argument("csv", help="CSV with r,g,b as first 3 columns (first row assumed header)")
    p.add_argument("--hue", action="store_true", help="Convert RGB -> hue and run circular stats & hue histograms")
    p.add_argument("--bins", type=int, default=36, help="Number of bins for histograms (default: 36). For hue this splits 360° into bins.")
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show(); useful when saving plots externally")
    p.add_argument("--list-terms", action="store_true", help="List unique color terms in 'gemini' and 'chatgpt' columns and exit")
    p.add_argument("--sort-terms", action="store_true", help="Sort listed color terms alphabetically instead of by frequency")
    args = p.parse_args()

    if args.list_terms:
        list_unique_color_terms(args.csv, sort_alpha=args.sort_terms)
        return

    rgb_data = read_rgb(args.csv)
    n = rgb_data.shape[0]

    if args.hue:
        # Convert RGB -> HSV hue (colorsys returns h in [0,1))
        hues = []
        for r, g, b in rgb_data:
            h, _, _ = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            hues.append(h * 360.0)
        hues = np.array(hues)

        mean_hue, R, circ_var, circ_std_deg = circular_stats_deg(hues)
        print(f"Hue stats (n={n}):")
        print(f"  mean angle = {mean_hue:.2f}°")
        print(f"  mean resultant length R = {R:.4f}")
        print(f"  circular variance = {circ_var:.4f}")
        print(f"  circular std (approx) = {circ_std_deg:.2f}°")

        # Chi-square test against uniformity
        nbins = args.bins
        counts, bin_edges = np.histogram(hues, bins=nbins, range=(0.0, 360.0))
        expected = np.full_like(counts, fill_value=n / nbins, dtype=float)
        stat, pval = chisquare(counts, f_exp=expected)
        print(f"Hue uniformity chi2 test (bins={nbins}): chi2={stat:.2f}, p={pval:.4f} (high p -> consistent with uniform)")

        # Plot linear histogram and polar rose
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.hist(hues, bins=nbins, range=(0, 360), color="purple", alpha=0.7)
        ax1.set_xlim(0, 360)
        ax1.set_xlabel("Hue (degrees)")
        ax1.set_ylabel("Count")
        ax1.set_title("Hue histogram")

        # Polar plot
        ax2 = fig.add_subplot(1, 2, 2, projection="polar")
        # Use bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        angles = np.deg2rad(bin_centers)
        widths = np.deg2rad(bin_edges[1] - bin_edges[0])
        # Normalize counts for bar height or use raw counts
        heights = counts
        # Color each bar by its hue
        cmap = matplotlib.colormaps["hsv"]
        colors = [cmap((bc % 360) / 360.0) for bc in bin_centers]
        bars = ax2.bar(angles, heights, width=widths, bottom=0.0, color=colors, edgecolor="k", alpha=0.8)
        ax2.set_title("Hue rose (polar histogram)")

        # Mark mean angle on polar plot
        mean_angle_rad = np.deg2rad(mean_hue)
        ax2.plot([mean_angle_rad, mean_angle_rad], [0, heights.max()], color="white", linewidth=2, linestyle="--", label=f"mean {mean_hue:.1f}°")
        ax2.legend(loc="upper right")

        plt.tight_layout()
        if not args.no_show:
            plt.show()

    else:
        # Original per-channel stats & plots
        print(f"RGB stats (n={n}):")
        means = rgb_data.mean(axis=0)
        stds = rgb_data.std(axis=0)

        for i, ch in enumerate("RGB"):
            counts, _ = np.histogram(rgb_data[:, i], bins=256, range=(0, 256))
            stat, p = chisquare(counts)
            print(f"  {ch}: mean={means[i]:.2f}, std={stds[i]:.2f}, chi2={stat:.2f}, p={p:.4f}")  # p >> 0.05 means uniform

        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for i, (ax, ch, color) in enumerate(zip(axes, "RGB", ["red", "green", "blue"])):
            ax.hist(rgb_data[:, i], bins=64, color=color, alpha=0.7)
            ax.set_title(f"{ch} channel")
            ax.set_xlim(0, 255)
        plt.tight_layout()
        if not args.no_show:
            plt.show()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        pairs = [("R", "G", 0, 1), ("R", "B", 0, 2), ("G", "B", 1, 2)]
        for ax, (a, b, i, j) in zip(axes, pairs):
            ax.hist2d(rgb_data[:, i], rgb_data[:, j], bins=64, cmap="viridis")
            ax.set_xlabel(a); ax.set_ylabel(b)
            ax.set_title(f"{a} vs {b}")
        plt.tight_layout()
        if not args.no_show:
            plt.show()

if __name__ == "__main__":
    main()
