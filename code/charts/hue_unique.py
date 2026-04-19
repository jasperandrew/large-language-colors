"""
unique_terms.py — Chart of unique color terms per hue bin.

Usage:
    python unique_terms.py data.csv [--n-bins 36] [--sigma 3.0]
                                    [--term-col color_term] [--output chart.png]
                                    [--font "Noto Sans CJK SC"]
                                    [--min-saturation 0.1] [--min-value 0.1]
"""

import argparse
import os
from functools import lru_cache
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.ndimage import gaussian_filter1d


# ── CJK font support (shared logic) ──────────────────────────────────────────

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
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in _CJK_FONT_CANDIDATES:
        if candidate in available:
            return candidate
    for font in font_manager.fontManager.ttflist:
        if any(kw in font.name.lower() for kw in _CJK_KEYWORDS):
            return font.name
    return None


def configure_font(font_spec: Optional[str]) -> None:
    if font_spec is None:
        font_spec = _find_cjk_font()
        if font_spec is None:
            return
    try:
        if os.path.exists(font_spec):
            font_manager.fontManager.addfont(font_spec)
            fp = font_manager.FontProperties(fname=font_spec)
            font_spec = fp.get_name()
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


# ── Chart builder ─────────────────────────────────────────────────────────────

def build_chart(
    csv_path: str,
    n_bins: int = 36,
    output_path: Optional[str] = None,
    term_col: str = "color_term",
    font: Optional[str] = None,
    sigma: float = 3.0,
    min_saturation: float = 0.1,
    min_value: float = 0.1,
) -> None:
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

    # ── Hue computation ───────────────────────────────────────────────────────
    df["hue"] = compute_hues(df)

    # ── Bin hues ──────────────────────────────────────────────────────────────
    bin_width = 360.0 / n_bins
    bins = np.arange(0, 360 + bin_width, bin_width)
    bin_labels = bins[:-1].astype(float)
    df["hue_bin"] = pd.cut(df["hue"], bins=bins, labels=bin_labels, right=False)
    df["hue_bin"] = df["hue_bin"].astype(float)

    # ── Count unique terms per bin ────────────────────────────────────────────
    unique_per_bin = (
        df.groupby("hue_bin", observed=True)[term_col]
        .nunique()
        .reindex(bin_labels, fill_value=0)
        .sort_index()
    )

    # ── Smooth ────────────────────────────────────────────────────────────────
    dense_step = bin_width / 10  # 10× denser than the bins
    x_dense = np.arange(0, 360, dense_step)
    y_interp = np.interp(x_dense, unique_per_bin.index.values, unique_per_bin.values.astype(float))

    sigma_samples = sigma / dense_step
    tiled = np.tile(y_interp, 3)
    smoothed = gaussian_filter1d(tiled, sigma=sigma_samples)
    y_smooth = smoothed[len(x_dense): 2 * len(x_dense)]
    y_smooth = np.clip(y_smooth, 0, None)

    # ── Font ──────────────────────────────────────────────────────────────────
    configure_font(font)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

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
    ax.fill_between(x_dense, 0, y_smooth, color="black", alpha=0.07, linewidth=0)

    # Raw bin values as faint dots for reference
    ax.scatter(
        unique_per_bin.index.values,
        unique_per_bin.values,
        s=18,
        color="black",
        alpha=0.25,
        zorder=2,
        linewidths=0,
    )

    # ── Axes styling ──────────────────────────────────────────────────────────
    ax.set_xlim(0, 360)
    ax.set_ylim(0, unique_per_bin.max() * 1.2 + 1)
    ax.set_xlabel("Hue (degrees)", color="black", fontsize=11)
    ax.set_ylabel("Unique color terms", color="black", fontsize=11)
    ax.set_title(
        f"Unique color terms per hue bin  ({n_bins} bins)",
        color="black",
        fontsize=13,
        pad=14,
    )
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    ax.set_xticks(np.arange(0, 361, 30))
    ax.yaxis.get_major_locator().set_params(integer=True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight") #, transparent=True)
        print(f"Saved to {output_path}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unique color terms per hue bin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv", help="CSV with columns: r, g, b, <term-col>")
    parser.add_argument("--n-bins", type=int, default=36)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--term-col", type=str, default="color_term")
    parser.add_argument("--font", type=str, default=None)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--min-saturation", type=float, default=0.1)
    parser.add_argument("--min-value", type=float, default=0.1)
    args = parser.parse_args()

    build_chart(
        csv_path=args.csv,
        n_bins=args.n_bins,
        output_path=args.output,
        term_col=args.term_col,
        font=args.font,
        sigma=args.sigma,
        min_saturation=args.min_saturation,
        min_value=args.min_value,
    )


if __name__ == "__main__":
    main()