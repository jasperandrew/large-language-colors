"""
entropy_lh.py — Shannon entropy of color terms over CIELAB L* × Hue space.

Usage:
    python entropy_lh.py data.csv [--n-bins 128] [--sigma 1.5]
                                  [--term-col color_term] [--output chart.png]
                                  [--font ...] [--bg white]
                                  [--min-saturation 0.0] [--min-value 0.0]
"""

import argparse
import os
from functools import lru_cache
from typing import Optional

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.ndimage import gaussian_filter

import tools.font as font_tools
import tools.color as color_tools


# ── Binning ───────────────────────────────────────────────────────────────────

def bin2d_entropy(x, y, terms, x_edges, y_edges):
    """
    Bin (x, y) points and compute per-cell Shannon entropy and sample count.

    Returns:
        entropy : 2D array (n_ybins × n_xbins)
        count   : 2D array (n_ybins × n_xbins)
    """
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    xi = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, nx - 1)
    yi = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, ny - 1)

    entropy = np.zeros((ny, nx), dtype=float)
    count   = np.zeros((ny, nx), dtype=int)

    cell_idx = yi * nx + xi
    order = np.argsort(cell_idx)
    cell_sorted = cell_idx[order]
    vals_sorted = terms[order]

    starts = np.searchsorted(cell_sorted, np.arange(ny * nx))
    ends   = np.searchsorted(cell_sorted, np.arange(ny * nx), side="right")

    for flat_i in range(ny * nx):
        s, e = starts[flat_i], ends[flat_i]
        if s == e:
            continue
        row, col = divmod(flat_i, nx)
        cell_terms = vals_sorted[s:e]
        count[row, col] = e - s
        _, term_counts = np.unique(cell_terms, return_counts=True)
        probs = term_counts / term_counts.sum()
        entropy[row, col] = -np.sum(probs * np.log2(probs + 1e-12))

    return entropy, count


# ── Chart builder ─────────────────────────────────────────────────────────────

def build_chart(
    csv_path: str,
    l_bins: int = 128,
    output_path: Optional[str] = None,
    term_col: str = "color_term",
    font: Optional[str] = None,
    sigma: float = 1.5,
    bg: str = "white",
    min_saturation: float = 0.01,
    min_value: float = 0.01,
    cmap: str = "magma",
    vmax: Optional[float] = None,
) -> None:
    """
    Plot Shannon entropy of color terms over CIELAB L* × Hue space.

    Parameters
    ----------
    csv_path        : CSV with columns r, g, b, <term_col>
    l_bins          : Grid resolution per axis (default 128)
    output_path     : Save path; None = interactive display
    term_col        : Color term column name
    font            : Font family or file path; None = auto-detect CJK
    sigma           : Gaussian smoothing sigma for the opacity mask
    bg              : Background color ('white' or '#1a1a1a')
    min_saturation  : HSV saturation threshold for achromatic filtering
    min_value       : HSV value threshold for achromatic filtering
    cmap            : Colormap for entropy (default: 'magma')
    vmax            : Fix the colormap maximum. If None, uses the per-file max.
                      Set to the same value across files for a matched scale.
    """
    fg = "black" if bg == "white" else "white"

    # ── Load & validate ───────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    missing = {"r", "g", "b", term_col} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    font_tools.configure_font(font)

    r = df["r"].to_numpy(dtype=float)
    g = df["g"].to_numpy(dtype=float)
    b = df["b"].to_numpy(dtype=float)

    # ── Optional achromatic filter ────────────────────────────────────────────
    if min_saturation > 0 or min_value > 0:
        sat, val = color_tools.rgb_to_sat_val(r.copy(), g.copy(), b.copy())
        mask = (sat >= min_saturation) & (val >= min_value)
        print(f"Filtered {(~mask).sum()} achromatic rows")
        df = df[mask].reset_index(drop=True)
        r, g, b = r[mask], g[mask], b[mask]

    terms = df[term_col].to_numpy()

    # ── CIELAB conversion ─────────────────────────────────────────────────────
    L, a_ch, b_ch = color_tools.rgb_to_lab(r.copy(), g.copy(), b.copy())
    hue = np.rad2deg(np.arctan2(b_ch, a_ch)) % 360

    # ── Bin & compute entropy ─────────────────────────────────────────────────
    hue_edges = np.linspace(0, 360, l_bins*3 + 1)
    L_edges   = np.linspace(0, 100, l_bins + 1)

    entropy, count = bin2d_entropy(hue, L, terms, hue_edges, L_edges)

    # ── Opacity mask: smooth count, scale to [0, 1] ───────────────────────────
    smoothed_count = gaussian_filter(count.astype(float), sigma=sigma)
    if smoothed_count.max() > 0:
        smoothed_count /= smoothed_count.max()
    alpha = np.sqrt(smoothed_count)  # sqrt for pleasing midtone boost

    # ── RGBA image ────────────────────────────────────────────────────────────
    file_max = entropy.max()
    norm = mcolors.Normalize(vmin=0, vmax=vmax if vmax is not None else file_max)
    rgba = plt.get_cmap(cmap)(norm(entropy))
    rgba[..., 3] *= alpha

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.imshow(
        rgba,
        origin="lower",
        extent=[0, 360, 0, 100],
        aspect="auto",
        interpolation="nearest",
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Shannon entropy (bits)", color=fg, fontsize=10)
    cb.ax.tick_params(colors=fg, labelsize=8)
    cb.outline.set_edgecolor("#aaaaaa" if bg == "white" else "#555555")

    # Axes
    ax.set_xlabel("Hue (°)", color=fg, fontsize=11)
    ax.set_ylabel("L* (lightness)", color=fg, fontsize=11)
    ax.set_title(
        f"Color term entropy — CIELAB L* × Hue  ({l_bins}×{l_bins*3} bins)",
        color=fg, fontsize=13, pad=12,
    )
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.tick_params(colors=fg, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa" if bg == "white" else "#555555")

    plt.tight_layout()

    print(f"Entropy max: {file_max:.4f} bits")

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight",
                    transparent=(bg != "white"), facecolor=bg)
        print(f"Saved to {output_path}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shannon entropy of color terms over CIELAB L* × Hue space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv")
    parser.add_argument("--l-bins", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--term-col", type=str, default="color_term")
    parser.add_argument("--font", type=str, default=None)
    parser.add_argument("--sigma", type=float, default=1.5,
                        help="Gaussian smoothing sigma for the opacity mask")
    parser.add_argument("--bg", type=str, default="white",
                        help="Background: 'white' or '#1a1a1a'")
    parser.add_argument("--cmap", type=str, default="magma",
                        help="Matplotlib colormap for entropy values")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Fix colormap maximum in bits. Use the same value "
                             "across files to get a matched scale.")
    parser.add_argument("--min-saturation", type=float, default=0.01)
    parser.add_argument("--min-value", type=float, default=0.01)
    args = parser.parse_args()

    build_chart(
        csv_path=args.csv,
        l_bins=args.l_bins,
        output_path=args.output,
        term_col=args.term_col,
        font=args.font,
        sigma=args.sigma,
        bg=args.bg,
        cmap=args.cmap,
        vmax=args.vmax,
        min_saturation=args.min_saturation,
        min_value=args.min_value,
    )


if __name__ == "__main__":
    main()