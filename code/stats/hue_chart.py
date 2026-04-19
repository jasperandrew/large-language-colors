"""
chart.py — Smoothed stacked-area hue chart from RGB + color-term CSV data.

Usage:
    python chart.py data.csv [--n-bins 36] [--sigma 3.0] [--top-n 10]
                             [--term-col color_term] [--output chart.png]
                             [--font "Noto Sans CJK SC"] [--dense-step 0.5]
                             [--min-saturation 0.1] [--min-value 0.1]
"""

import argparse
import os
from functools import lru_cache
from typing import Optional

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.ndimage import gaussian_filter1d


# ── CJK font support ──────────────────────────────────────────────────────────

def needs_cjk(texts: list[str]) -> bool:
    """Return True if any string in texts contains CJK or Hangul characters."""
    for text in texts:
        for ch in text:
            cp = ord(ch)
            if (
                0x1100 <= cp <= 0x11FF    # Hangul Jamo
                or 0xAC00 <= cp <= 0xD7AF  # Hangul Syllables
                or 0x3040 <= cp <= 0x30FF  # Hiragana / Katakana
                or 0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
                or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
                or 0x20000 <= cp <= 0x2A6DF  # CJK Extension B
                or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
            ):
                return True
    return False


_CJK_FONT_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK KR", "Noto Sans CJK TC", "Noto Sans CJK JP",
    "PingFang SC", "Apple SD Gothic Neo", "Hiragino Sans",
    "Microsoft YaHei", "Malgun Gothic", "MS Gothic",
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "UnDotum", "NanumGothic",
    "SimHei", "SimSun", "NSimSun",
]

_CJK_KEYWORDS = ("cjk", "noto", "gothic", "hei", "yuan", "ming", "dotum",
                  "gulim", "batang", "nanum", "malgun", "pingfang", "hiragino")


@lru_cache(maxsize=1)
def _find_cjk_font() -> Optional[str]:
    """
    Search the system font registry for a CJK-capable family.
    Result is cached — font manager scans can be slow.
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in _CJK_FONT_CANDIDATES:
        if candidate in available:
            return candidate
    for font in font_manager.fontManager.ttflist:
        if any(kw in font.name.lower() for kw in _CJK_KEYWORDS):
            return font.name
    return None


def configure_font(font_spec: Optional[str], labels: Optional[list[str]] = None) -> None:
    """
    Set matplotlib's active font.

    - font file path  → register and activate it
    - font family name → activate directly
    - None            → auto-detect a CJK font if labels require one
    """
    if font_spec is None:
        if labels and needs_cjk(labels):
            font_spec = _find_cjk_font()
            if font_spec:
                print(f"Auto-detected CJK font: '{font_spec}'")
            else:
                print(
                    "Warning: CJK/Hangul characters found in labels but no CJK font "
                    "is installed. Install a Noto CJK font for correct rendering."
                )
                return
        else:
            return

    try:
        if os.path.exists(font_spec):
            font_manager.fontManager.addfont(font_spec)
            fp = font_manager.FontProperties(fname=font_spec)
            font_spec = fp.get_name()
            print(f"Loaded font from file: '{font_spec}'")

        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = [font_spec] + list(
            mpl.rcParams.get("font.sans-serif", [])
        )
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception as exc:
        print(f"Warning: could not set font '{font_spec}': {exc}")


# ── Color helpers ─────────────────────────────────────────────────────────────

def compute_hues(df: pd.DataFrame) -> np.ndarray:
    """
    Vectorized HSV hue for a DataFrame with r, g, b columns (0–255).
    Returns an array of hues in degrees [0, 360).
    """
    r = df["r"].to_numpy(dtype=float) / 255
    g = df["g"].to_numpy(dtype=float) / 255
    b = df["b"].to_numpy(dtype=float) / 255

    cmax = np.maximum.reduce([r, g, b])
    cmin = np.minimum.reduce([r, g, b])
    delta = cmax - cmin

    hue = np.zeros(len(r))

    mask = (cmax == r) & (delta > 0)
    hue[mask] = (60 * ((g[mask] - b[mask]) / delta[mask])) % 360

    mask = (cmax == g) & (delta > 0)
    hue[mask] = 60 * ((b[mask] - r[mask]) / delta[mask] + 2)

    mask = (cmax == b) & (delta > 0)
    hue[mask] = 60 * ((r[mask] - g[mask]) / delta[mask] + 4)

    return hue


def compute_saturation_value(df: pd.DataFrame):
    """Return (saturation, value) arrays for HSV, vectorized, from 0–255 RGB."""
    r = df["r"].to_numpy(dtype=float) / 255
    g = df["g"].to_numpy(dtype=float) / 255
    b = df["b"].to_numpy(dtype=float) / 255
    cmax = np.maximum.reduce([r, g, b])
    cmin = np.minimum.reduce([r, g, b])
    delta = cmax - cmin
    saturation = np.where(cmax > 0, delta / cmax, 0.0)
    return saturation, cmax  # cmax == HSV value


def compute_term_colors(df: pd.DataFrame, term_col: str) -> dict[str, tuple]:
    """
    Mean RGB per color term as (r, g, b) in [0, 1].
    Vectorized via groupby — much faster than row-wise apply.
    """
    means = df.groupby(term_col)[["r", "g", "b"]].mean() / 255
    return {term: (row["r"], row["g"], row["b"]) for term, row in means.iterrows()}


# ── Core chart builder ────────────────────────────────────────────────────────

def build_chart(
    csv_path: str,
    n_bins: int = 36,
    output_path: Optional[str] = None,
    text_color: str = "black",
    term_col: str = "color_term",
    top_n: Optional[int] = None,
    font: Optional[str] = None,
    sigma: float = 3.0,
    dense_step: float = 0.5,
    min_saturation: float = 0.1,
    min_value: float = 0.1,
) -> None:
    """
    Build and display (or save) a smoothed stacked-area hue chart.

    Parameters
    ----------
    csv_path        : Path to CSV with columns: r, g, b, <term_col>
    n_bins          : Number of hue bins for initial aggregation
    output_path     : Save path; if None, displays interactively
    text_color      : Color for legend, axis, and label text
    term_col        : Name of the color-term column in the CSV
    top_n           : Restrict to the N most frequent terms (None = all)
    font            : Font family name or file path; None = auto-detect CJK
    sigma           : Gaussian smoothing width in degrees
    dense_step      : Step size of the dense interpolation grid in degrees
    min_saturation  : Rows with HSV saturation below this are excluded.
                      Achromatic colors (grays/blacks) have hue=0 by convention,
                      which incorrectly inflates the red bin without this filter.
    min_value       : Rows with HSV value (brightness) below this are excluded.
    """
    # ── Load & validate ───────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    missing = {"r", "g", "b", term_col} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # ── Filter achromatic colors ──────────────────────────────────────────────
    saturation, value = compute_saturation_value(df)
    chromatic = (saturation >= min_saturation) & (value >= min_value)
    n_dropped = (~chromatic).sum()
    if n_dropped:
        print(f"Filtered {n_dropped} achromatic rows "
              f"(sat < {min_saturation} or val < {min_value})")
    df = df[chromatic].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No chromatic rows remain after filtering. "
            "Try lowering --min-saturation or --min-value."
        )

    # ── Hue & term colors (both vectorized) ───────────────────────────────────
    df["hue"] = compute_hues(df)
    term_colors = compute_term_colors(df, term_col)  # before top-N filter

    # ── Bin hues ──────────────────────────────────────────────────────────────
    bin_width = 360.0 / n_bins
    bins = np.arange(0, 360 + bin_width, bin_width)
    bin_labels = bins[:-1].astype(float)
    df["hue_bin"] = pd.cut(df["hue"], bins=bins, labels=bin_labels, right=False)
    df["hue_bin"] = df["hue_bin"].astype(float)

    # ── Counts & proportions ──────────────────────────────────────────────────
    counts = (
        df.groupby(["hue_bin", term_col], observed=True)
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("hue_bin", observed=True)["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals

    # ── Optional top-N filter ─────────────────────────────────────────────────
    if top_n is not None and top_n > 0:
        top_terms = df[term_col].value_counts().nlargest(top_n).index.tolist()
        counts = counts[counts[term_col].isin(top_terms)].copy()
        totals = counts.groupby("hue_bin", observed=True)["count"].transform("sum")
        counts["proportion"] = (counts["count"] / totals).fillna(0)
        term_colors = {k: v for k, v in term_colors.items() if k in top_terms}

    # ── Pivot ─────────────────────────────────────────────────────────────────
    pivot = (
        counts.pivot_table(
            index="hue_bin", columns=term_col, values="proportion", aggfunc="sum"
        )
        .fillna(0)
        .reindex(bin_labels, fill_value=0)
        .sort_index()
    )

    # ── Sort terms by their mean hue for a logical legend order ──────────────
    def mean_hue(term: str) -> float:
        r, g, b = term_colors[term]
        # circular mean of hue, weighted by per-row counts
        subset = df[df[term_col] == term]["hue"].to_numpy()
        if len(subset) == 0:
            return 0.0
        rad = np.deg2rad(subset)
        return np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360
 
    color_terms = sorted(pivot.columns.tolist(), key=mean_hue)
    print(color_terms)
    pivot = pivot[color_terms]

    # ── Font configuration ────────────────────────────────────────────────────
    configure_font(font, labels=color_terms)

    # ── Smooth: upsample → Gaussian (circular) → renormalize ─────────────────
    x_dense = np.arange(0, 360, dense_step)
    n_dense = len(x_dense)
    n_terms = len(color_terms)
    dense_data = np.empty((n_dense, n_terms))

    for i, term in enumerate(color_terms):
        dense_data[:, i] = np.interp(x_dense, pivot.index.values, pivot[term].values)

    # Circular Gaussian smoothing (tile × 3, smooth, take middle)
    sigma_samples = sigma / dense_step
    for i in range(n_terms):
        tiled = np.tile(dense_data[:, i], 3)
        smoothed = gaussian_filter1d(tiled, sigma=sigma_samples)
        dense_data[:, i] = smoothed[n_dense: 2 * n_dense]

    np.clip(dense_data, 0, None, out=dense_data)
    row_sums = dense_data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    dense_data /= row_sums

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    bottom = np.zeros(n_dense)
    for i, term in enumerate(color_terms):
        color = term_colors.get(term, (0.5, 0.5, 0.5))
        top = bottom + dense_data[:, i]
        ax.fill_between(x_dense, bottom, top, color=color, linewidth=0)
        bottom = top

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=term_colors.get(t, (0.5, 0.5, 0.5)), label=t)
        for t in color_terms
    ]
    legend = ax.legend(
        handles=patches,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        framealpha=0,
        edgecolor="none",
        labelcolor=text_color,
        fontsize=8,
        title="Color term",
        title_fontsize=9,
    )
    legend.get_title().set_color(text_color)

    # ── Axes styling ──────────────────────────────────────────────────────────
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Hue (degrees)", color=text_color, fontsize=11)
    ax.set_ylabel("Proportion", color=text_color, fontsize=11)
    ax.set_title(
        f"Color term distribution by hue  (top {top_n} terms, {n_bins} bins)",
        color=text_color,
        fontsize=13,
        pad=14,
    )
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    ax.set_xticks(np.arange(0, 361, 30))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"Saved to {output_path}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoothed stacked-area hue chart from an RGB + color-term CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv", help="CSV with columns: r, g, b, <term-col>")
    parser.add_argument("--n-bins", type=int, default=36,
                        help="Number of hue bins for initial aggregation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (omit to display interactively)")
    parser.add_argument("--text-color", type=str, default="black",
                        help="Color for legend, axis, and label text")
    parser.add_argument("--term-col", type=str, default="color_term",
                        help="CSV column containing color terms")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Only show the N most frequent color terms")
    parser.add_argument("--font", type=str, default=None,
                        help="Font family name or file path (auto-detects CJK if omitted)")
    parser.add_argument("--sigma", type=float, default=3.0,
                        help="Gaussian smoothing width in degrees")
    parser.add_argument("--dense-step", type=float, default=0.1,
                        help="Dense grid resolution in degrees")
    parser.add_argument("--min-saturation", type=float, default=0.1,
                        help="HSV saturation threshold; rows below are excluded as achromatic")
    parser.add_argument("--min-value", type=float, default=0.1,
                        help="HSV value (brightness) threshold; rows below are excluded")
    args = parser.parse_args()

    build_chart(
        csv_path=args.csv,
        n_bins=args.n_bins,
        output_path=args.output,
        text_color=args.text_color,
        term_col=args.term_col,
        top_n=args.top_n,
        font=args.font,
        sigma=args.sigma,
        dense_step=args.dense_step,
        min_saturation=args.min_saturation,
        min_value=args.min_value,
    )


if __name__ == "__main__":
    main()