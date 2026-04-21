"""
unique_terms.py — Chart of unique color terms per hue bin.

Usage:
    python unique_terms.py data.csv [--n-bins 36] [--sigma 3.0]
                                    [--term-col color_term] [--output chart.png]
                                    [--font "Noto Sans CJK SC"]
                                    [--min-sat 0.1] [--min-val 0.1]
"""

import argparse
import os
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import tools.color
import tools.data
import tools.font


def build_chart(
    csv_path: str,
    data_lang: str,
    data_src: str,
    term_col: str,
    n_bins: int = 36,
    output_path: Optional[str] = None,
    font: Optional[str] = None,
    sigma: float = 3.0,
    ymax: Optional[float] = None,
    min_sat: float = 0.1,
    min_val: float = 0.1,
) -> None:

    df = tools.color.filter_achromatic(
        tools.data.load_and_validate(csv_path, term_col),
        min_sat, min_val
    )

    # Hue computation 
    df["hue"] = tools.color.vec_rgb_to_h(df)

    # Bin hues 
    bin_width = 360.0 / n_bins
    bins = np.arange(0, 360 + bin_width, bin_width)
    bin_labels = bins[:-1].astype(float)
    df["hue_bin"] = pd.cut(df["hue"], bins=bins, labels=bin_labels, right=False)
    df["hue_bin"] = df["hue_bin"].astype(float)

    # Count unique terms per bin 
    unique_per_bin = (
        df.groupby("hue_bin", observed=True)[term_col]
        .nunique()
        .reindex(bin_labels, fill_value=0)
        .sort_index()
    )

    # Smooth 
    dense_step = bin_width / 10  # 10× denser than the bins
    x_dense = np.arange(0, 360, dense_step)
    y_interp = np.interp(x_dense, unique_per_bin.index.values, unique_per_bin.values.astype(float))

    sigma_samples = sigma / dense_step
    tiled = np.tile(y_interp, 3)
    smoothed = gaussian_filter1d(tiled, sigma=sigma_samples)
    y_smooth = smoothed[len(x_dense): 2 * len(x_dense)]
    y_smooth = np.clip(y_smooth, 0, None)

    # Plot
    bg = "white"
    fg = "black"
    tools.font.configure_font(font)

    fig, ax = plt.subplots(figsize=(9, 1))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # Draw a rainbow-colored line by rendering many thin segments
    from matplotlib.collections import LineCollection
    points = np.array([x_dense, y_smooth]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    hue_colors = [
        tuple(c) for c in plt.cm.hsv(x_dense[:-1] / 360)
    ]
    lc = LineCollection(segments, colors=hue_colors, linewidth=2.5, zorder=3)
    ax.add_collection(lc)

    # Filled area beneath the line, in a neutral color
    # ax.fill_between(x_dense, 0, y_smooth, color=fg, alpha=0.07, linewidth=0)

    # Raw bin values as faint dots for reference
    # ax.scatter(
    #     unique_per_bin.index.values,
    #     unique_per_bin.values,
    #     s=18,
    #     color=fg,
    #     alpha=0.25,
    #     zorder=2,
    #     linewidths=0,
    # )

    # Axes styling 
    ax.set_xlim(0, 360)
    ax.set_ylim(0, ymax if ymax is not None else unique_per_bin.max() * 1.1)
    ax.set_xlabel("Hue (degrees)", color=fg, fontsize=11)
    ax.set_ylabel("Unique color terms", color=fg, fontsize=11)
    lang_str = "" if data_lang == None else data_lang.upper() + ", "
    src_str = "" if data_src == None else data_src + ", "
    ax.set_title(
        f"Unique color terms per hue bin  ({lang_str}{src_str}{n_bins} bins)",
        color=fg,
        fontsize=13,
        pad=14,
    )
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
    ax.set_xticks(np.arange(0, 361, 30))
    ax.yaxis.get_major_locator().set_params(integer=True)

    plt.tight_layout()

    print(f"Max unique terms: {unique_per_bin.max()}")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight") #, transparent=True)
        print(f"Saved to {output_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unique color terms per hue bin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv",                                    help="CSV with columns: r, g, b, <term-col>")
    parser.add_argument("data_lang",    type=str,                 help="Language of color terms (e.g. en, ko, zh)")
    parser.add_argument("data_src",     type=str,                 help="Source of color terms (e.g. human, gpt, gemini)")
    parser.add_argument("term_col",     type=str,                 help="CSV column containing color terms")
    parser.add_argument("--n-bins",     type=int,   default=36,   help="Number of hue bins for initial aggregation")
    parser.add_argument("--output",     type=str,   default=None, help="Output image path (omit to display interactively)")
    parser.add_argument("--font",       type=str,   default=None, help="Font family name or file path (auto-detects CJK if omitted)")
    parser.add_argument("--sigma",      type=float, default=3.0,  help="Gaussian smoothing width in degrees")
    parser.add_argument("--ymax",       type=float, default=None, help="Fixed maximum y-axis value")
    parser.add_argument("--min-sat",    type=float, default=0.1,  help="HSV saturation threshold")
    parser.add_argument("--min-val",    type=float, default=0.1,  help="HSV value threshold")
    args = parser.parse_args()

    build_chart(
        args.csv,
        args.data_lang,
        args.data_src,
        args.term_col,
        n_bins=args.n_bins,
        output_path=args.output,
        font=args.font,
        sigma=args.sigma,
        ymax=args.ymax,
        min_sat=args.min_sat,
        min_val=args.min_val,
    )


if __name__ == "__main__":
    main()