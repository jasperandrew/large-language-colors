"""
entropy_lh.py — Shannon entropy of color terms over CIELAB L* × Hue space.

Usage:
    python entropy_lh.py data.csv [--n-bins 128] [--sigma 1.5]
                                  [--term-col color_term] [--output chart.png]
                                  [--font ...]
                                  [--min-sat 0.0] [--min-val 0.0]
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

import tools.color
import tools.data
import tools.font


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


def build_chart(
    csv_path: str,
    data_lang: str,
    data_src: str,
    term_col: str,
    l_bins: int = 128,
    output_path: Optional[str] = None,
    font: Optional[str] = None,
    sigma: float = 1.5,
    min_sat: float = 0.01,
    min_val: float = 0.01,
    cmap: str = "viridis",
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
    min_sat  : HSV saturation threshold for achromatic filtering
    min_val       : HSV value threshold for achromatic filtering
    cmap            : Colormap for entropy (default: 'magma')
    vmax            : Fix the colormap maximum. If None, uses the per-file max.
                      Set to the same value across files for a matched scale.
    """

    df = tools.color.filter_achromatic(
        tools.data.load_and_validate(csv_path, term_col),
        min_sat, min_val
    )

    terms = df[term_col].to_numpy()

    # CIELAB conversion 
    L, a, b = tools.color.vec_rgb_to_lab(df)
    hue = np.rad2deg(np.arctan2(b, a)) % 360

    n_bins = l_bins*5

    # Bin & compute entropy 
    hue_edges = np.linspace(0, 360, n_bins + 1)
    L_edges   = np.linspace(0, 100, l_bins + 1)

    entropy, count = bin2d_entropy(hue, L, terms, hue_edges, L_edges)

    # Opacity mask: smooth count, scale to [0, 1] 
    smoothed_count = gaussian_filter(count.astype(float), sigma=sigma)
    if smoothed_count.max() > 0:
        smoothed_count /= smoothed_count.max()
    alpha = np.sqrt(smoothed_count)  # sqrt for pleasing midtone boost

    # RGBA image 
    file_max = entropy.max()
    norm = mcolors.Normalize(vmin=0, vmax=vmax if vmax is not None else file_max)
    rgba = plt.get_cmap(cmap)(norm(entropy))
    rgba[..., 3] *= alpha

    # Plot 
    bg = "white"
    fg = "black"
    tools.font.configure_font(font)

    fig, ax = plt.subplots(figsize=(15, 3))
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
    cb.outline.set_edgecolor("#aaaaaa")

    # Axes
    ax.set_xlabel("Hue (°)", color=fg, fontsize=11)
    ax.set_ylabel("L* (lightness)", color=fg, fontsize=11)
    lang_str = "" if data_lang == None else data_lang.upper() + ", "
    src_str = "" if data_src == None else data_src + ", "
    ax.set_title(
        f"Color term entropy - CIELAB L* × Hue  ({lang_str}{src_str}{l_bins}×{n_bins} bins)",
        color=fg, fontsize=13, pad=12,
    )
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.tick_params(colors=fg, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")

    plt.tight_layout()

    print(f"Entropy max: {file_max:.4f} bits")

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shannon entropy of color terms over CIELAB L* × Hue space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv",                                         help="CSV with columns: r, g, b, <term-col>")
    parser.add_argument("data_lang",    type=str,                      help="Language of color terms (e.g. en, ko, zh)")
    parser.add_argument("data_src",     type=str,                      help="Source of color terms (e.g. human, gpt, gemini)")
    parser.add_argument("term_col",     type=str,                      help="CSV column containing color terms")
    parser.add_argument("--l-bins",     type=int,   default=36,        help="Number of hue bins for initial aggregation")
    parser.add_argument("--output",     type=str,   default=None,      help="Output image path (omit to display interactively)")
    parser.add_argument("--font",       type=str,   default=None,      help="Font family name or file path (auto-detects CJK if omitted)")
    parser.add_argument("--sigma",      type=float, default=3.0,       help="Gaussian smoothing width in degrees")
    parser.add_argument("--cmap",       type=str,   default="viridis", help="Matplotlib colormap for entropy values")
    parser.add_argument("--vmax",       type=float, default=None,      help="Fixed colormap maximum in bits")
    parser.add_argument("--min-sat",    type=float, default=0.1,       help="HSV saturation threshold")
    parser.add_argument("--min-val",    type=float, default=0.1,       help="HSV value threshold")
    args = parser.parse_args()

    build_chart(
        args.csv,
        args.data_lang,
        args.data_src,
        args.term_col,
        l_bins=args.l_bins,
        output_path=args.output,
        font=args.font,
        sigma=args.sigma,
        cmap=args.cmap,
        vmax=args.vmax,
        min_sat=args.min_sat,
        min_val=args.min_val,
    )


if __name__ == "__main__":
    main()