"""
color_space_map.py — 2D color space maps showing term richness across RGB/CIELAB space.

Produces a 3×3 grid:
  Rows:    HSV (Hue × Saturation) | CIELAB (a* × b*) | CIELAB (L* × Hue)
  Columns: Unique terms | Shannon entropy | Most common term

Sparse regions fade out proportionally to sample count.

Usage:
    python color_space_map.py data.csv [--n-bins 64] [--term-col color_term]
                                       [--output map.png] [--font ...] [--l-slice 50]
                                       [--min-saturation 0.0] [--min-value 0.0]
                                       [--sigma 1.0]
"""

import argparse
import os
from functools import lru_cache
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.ndimage import gaussian_filter


# ── CJK font support ──────────────────────────────────────────────────────────

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
    for c in _CJK_FONT_CANDIDATES:
        if c in available:
            return c
    for f in font_manager.fontManager.ttflist:
        if any(kw in f.name.lower() for kw in _CJK_KEYWORDS):
            return f.name
    return None


def configure_font(font_spec: Optional[str]) -> None:
    if font_spec is None:
        font_spec = _find_cjk_font()
        if font_spec is None:
            return
    try:
        if os.path.exists(font_spec):
            font_manager.fontManager.addfont(font_spec)
            font_spec = font_manager.FontProperties(fname=font_spec).get_name()
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = [font_spec] + list(mpl.rcParams.get("font.sans-serif", []))
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception as exc:
        print(f"Warning: could not set font '{font_spec}': {exc}")


# ── Color space conversions ───────────────────────────────────────────────────

def rgb_to_hsv_components(r, g, b):
    """Vectorized RGB (0–255) → HSV. Returns hue [0,360), sat [0,1], val [0,1]."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    cmax = np.maximum.reduce([r, g, b])
    cmin = np.minimum.reduce([r, g, b])
    delta = cmax - cmin

    hue = np.zeros(len(r))
    m = (cmax == r) & (delta > 0)
    hue[m] = (60 * ((g[m] - b[m]) / delta[m])) % 360
    m = (cmax == g) & (delta > 0)
    hue[m] = 60 * ((b[m] - r[m]) / delta[m] + 2)
    m = (cmax == b) & (delta > 0)
    hue[m] = 60 * ((r[m] - g[m]) / delta[m] + 4)

    sat = np.where(cmax > 0, delta / cmax, 0.0)
    return hue, sat, cmax  # cmax == value


def rgb_to_lab(r, g, b):
    """Vectorized RGB (0–255) → CIELAB (D65)."""
    # Linearize sRGB
    rgb = np.stack([r, g, b], axis=1) / 255.0
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] /= 12.92

    # sRGB → XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = rgb @ M.T

    # Normalize by D65 white point
    xyz /= np.array([0.95047, 1.00000, 1.08883])

    # XYZ → Lab
    eps, kap = 0.008856, 903.3
    fx = np.where(xyz > eps, xyz ** (1/3), (kap * xyz + 16) / 116)
    L = 116 * fx[:, 1] - 16
    a = 500 * (fx[:, 0] - fx[:, 1])
    b_ch = 200 * (fx[:, 1] - fx[:, 2])
    return L, a, b_ch


# ── Binning & metrics ─────────────────────────────────────────────────────────

def bin2d(x, y, values, x_edges, y_edges):
    """
    Bin (x, y) data into a 2D grid and compute per-cell metrics.

    Returns a dict of 2D arrays (shape: n_ybins × n_xbins):
      - count       : number of samples
      - n_unique    : number of unique color terms
      - entropy     : Shannon entropy of term distribution
      - modal_color : RGBA of the most common term's representative color
      - modal_alpha : opacity weight for modal_color (count-based)
    """
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    xi = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, nx - 1)
    yi = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, ny - 1)

    count     = np.zeros((ny, nx), dtype=int)
    n_unique  = np.zeros((ny, nx), dtype=float)
    entropy   = np.zeros((ny, nx), dtype=float)
    modal_r   = np.full((ny, nx), 0.5)
    modal_g   = np.full((ny, nx), 0.5)
    modal_b   = np.full((ny, nx), 0.5)

    # Group by cell
    cell_idx = yi * nx + xi
    order = np.argsort(cell_idx)
    cell_sorted = cell_idx[order]
    vals_sorted = values[order]

    starts = np.searchsorted(cell_sorted, np.arange(ny * nx))
    ends   = np.searchsorted(cell_sorted, np.arange(ny * nx), side="right")

    for flat_i in range(ny * nx):
        s, e = starts[flat_i], ends[flat_i]
        if s == e:
            continue
        row, col = divmod(flat_i, nx)
        cell_vals = vals_sorted[s:e]
        count[row, col] = e - s

        terms, term_counts = np.unique(cell_vals, return_counts=True)
        n_unique[row, col] = len(terms)

        probs = term_counts / term_counts.sum()
        entropy[row, col] = -np.sum(probs * np.log2(probs + 1e-12))

        modal_term = terms[np.argmax(term_counts)]
        modal_r[row, col], modal_g[row, col], modal_b[row, col] = \
            term_rgb_lookup.get(modal_term, (0.5, 0.5, 0.5))

    return {
        "count":       count,
        "n_unique":    n_unique,
        "entropy":     entropy,
        "modal_r":     modal_r,
        "modal_g":     modal_g,
        "modal_b":     modal_b,
    }


# ── Plotting helpers ──────────────────────────────────────────────────────────

def alpha_from_count(count: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Convert sample counts to opacity values.
    Smooth the count map, then scale so the max = 1.
    Sparse cells fade out gracefully.
    """
    smoothed = gaussian_filter(count.astype(float), sigma=sigma)
    if smoothed.max() > 0:
        smoothed /= smoothed.max()
    return np.sqrt(smoothed)  # sqrt gives a pleasing midtone boost


def apply_alpha(rgba: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Multiply the alpha channel of an RGBA image by a 2D alpha mask."""
    out = rgba.copy()
    out[..., 3] *= alpha
    return out


def scalar_to_rgba(data: np.ndarray, cmap: str, vmin=None, vmax=None) -> np.ndarray:
    """Map a 2D scalar array to RGBA using a colormap."""
    norm = mcolors.Normalize(vmin=vmin or np.nanmin(data),
                             vmax=vmax or np.nanmax(data))
    return plt.get_cmap(cmap)(norm(data))


def modal_to_rgba(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Assemble per-cell modal term RGB into an RGBA image (alpha=1)."""
    rgba = np.ones((*r.shape, 4))
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    return rgba


def style_ax(ax, xlabel, ylabel, title, bg="white"):
    fg = "black" if bg == "white" else "white"
    ax.set_facecolor(bg)
    ax.set_xlabel(xlabel, color=fg, fontsize=8)
    ax.set_ylabel(ylabel, color=fg, fontsize=8)
    ax.set_title(title, color=fg, fontsize=9, pad=6)
    ax.tick_params(colors=fg, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa" if bg == "white" else "#555555")


# ── Main ──────────────────────────────────────────────────────────────────────

# Module-level lookup used inside bin2d (set before calling)
term_rgb_lookup: dict = {}


def build_chart(
    csv_path: str,
    n_bins: int = 64,
    output_path: Optional[str] = None,
    term_col: str = "color_term",
    font: Optional[str] = None,
    sigma: float = 1.0,
    l_slice: float = 50.0,
    min_saturation: float = 0.0,
    min_value: float = 0.0,
    bg: str = "white",
) -> None:
    global term_rgb_lookup

    # ── Load ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    missing = {"r", "g", "b", term_col} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    configure_font(font)

    r = df["r"].to_numpy(dtype=float)
    g = df["g"].to_numpy(dtype=float)
    b = df["b"].to_numpy(dtype=float)

    # ── Optional achromatic filter ────────────────────────────────────────────
    if min_saturation > 0 or min_value > 0:
        hue, sat, val = rgb_to_hsv_components(r.copy(), g.copy(), b.copy())
        mask = (sat >= min_saturation) & (val >= min_value)
        print(f"Filtered {(~mask).sum()} achromatic rows")
        df, r, g, b = df[mask].reset_index(drop=True), r[mask], g[mask], b[mask]

    terms = df[term_col].to_numpy()

    # ── Build term → representative RGB lookup ────────────────────────────────
    means = df.groupby(term_col)[["r", "g", "b"]].mean() / 255
    term_rgb_lookup = {t: (row.r, row.g, row.b) for t, row in means.iterrows()}

    # ── Compute color space coordinates ───────────────────────────────────────
    hue, sat, val = rgb_to_hsv_components(r.copy(), g.copy(), b.copy())
    L, a_ch, b_ch = rgb_to_lab(r.copy(), g.copy(), b.copy())
    lab_hue = np.rad2deg(np.arctan2(b_ch, a_ch)) % 360

    # ── Grid edges ────────────────────────────────────────────────────────────
    hue_edges = np.linspace(0, 360, n_bins + 1)
    sat_edges = np.linspace(0, 1,   n_bins + 1)
    a_edges   = np.linspace(-128, 128, n_bins + 1)
    b_edges   = np.linspace(-128, 128, n_bins + 1)
    L_edges   = np.linspace(0, 100,  n_bins + 1)

    # ── Compute all three grids ───────────────────────────────────────────────
    grid_hsv  = bin2d(hue, sat,    terms, hue_edges, sat_edges)
    grid_lab  = bin2d(a_ch, b_ch,  terms, a_edges,   b_edges)
    grid_lh   = bin2d(lab_hue, L,  terms, hue_edges, L_edges)

    # ── Figure ────────────────────────────────────────────────────────────────
    fg = "black" if bg == "white" else "white"
    fig, axes = plt.subplots(
        3, 3,
        figsize=(14, 10),
        facecolor=bg,
    )
    fig.patch.set_facecolor(bg)

    row_labels = ["HSV: Hue × Saturation", "CIELAB: a* × b*", "CIELAB: L* × Hue"]
    col_labels = ["Unique terms", "Shannon entropy", "Most common term"]

    def draw(ax, grid, metric, x_edges, y_edges, xlabel, ylabel, title):
        alpha = alpha_from_count(grid["count"], sigma=sigma)
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        if metric == "n_unique":
            rgba = scalar_to_rgba(grid["n_unique"], "plasma")
            rgba = apply_alpha(rgba, alpha)
            im = ax.imshow(rgba, origin="lower", extent=extent,
                           aspect="auto", interpolation="bilinear")
            # Colorbar
            sm = plt.cm.ScalarMappable(
                cmap="plasma",
                norm=mcolors.Normalize(0, grid["n_unique"].max())
            )
            cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cb.ax.tick_params(colors=fg, labelsize=6)
            cb.outline.set_edgecolor("#aaaaaa" if bg == "white" else "#555555")

        elif metric == "entropy":
            rgba = scalar_to_rgba(grid["entropy"], "viridis")
            rgba = apply_alpha(rgba, alpha)
            ax.imshow(rgba, origin="lower", extent=extent,
                      aspect="auto", interpolation="bilinear")
            sm = plt.cm.ScalarMappable(
                cmap="viridis",
                norm=mcolors.Normalize(0, grid["entropy"].max())
            )
            cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cb.ax.tick_params(colors=fg, labelsize=6)
            cb.outline.set_edgecolor("#aaaaaa" if bg == "white" else "#555555")

        elif metric == "modal":
            rgba = modal_to_rgba(grid["modal_r"], grid["modal_g"], grid["modal_b"])
            rgba = apply_alpha(rgba, alpha)
            ax.imshow(rgba, origin="lower", extent=extent,
                      aspect="auto", interpolation="bilinear")

        style_ax(ax, xlabel, ylabel, title, bg=bg)

    metrics = ["n_unique", "entropy", "modal"]

    configs = [
        # (grid,      x_edges,    y_edges,    xlabel,    ylabel,        row_label)
        (grid_hsv,  hue_edges,  sat_edges,  "Hue (°)", "Saturation",  row_labels[0]),
        (grid_lab,  a_edges,    b_edges,    "a*",      "b*",          row_labels[1]),
        (grid_lh,   hue_edges,  L_edges,    "Hue (°)", "L*",          row_labels[2]),
    ]

    for row, (grid, x_edges, y_edges, xlabel, ylabel, row_label) in enumerate(configs):
        for col, metric in enumerate(metrics):
            title = col_labels[col] if row == 0 else ""
            draw(axes[row][col], grid, metric, x_edges, y_edges, xlabel, ylabel, title)
            if col == 0:
                axes[row][col].set_ylabel(f"{row_label}\n{ylabel}", color=fg, fontsize=8)

    # Row/column header annotations
    for col, label in enumerate(col_labels):
        axes[0][col].set_title(label, color=fg, fontsize=10, pad=8, fontweight="bold")

    plt.suptitle("Color term coverage across color spaces", color=fg, fontsize=13, y=1.01)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    transparent=(bg != "white"), facecolor=bg)
        print(f"Saved to {output_path}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="2D color space maps of term richness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv", help="CSV with columns: r, g, b, <term-col>")
    parser.add_argument("--n-bins", type=int, default=64,
                        help="Grid resolution per axis")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--term-col", type=str, default="color_term")
    parser.add_argument("--font", type=str, default=None)
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Gaussian smoothing sigma for the opacity mask")
    parser.add_argument("--bg", type=str, default="white",
                        help="Background color: 'white' or '#1a1a1a'")
    parser.add_argument("--min-saturation", type=float, default=0.0)
    parser.add_argument("--min-value", type=float, default=0.0)
    args = parser.parse_args()

    build_chart(
        csv_path=args.csv,
        n_bins=args.n_bins,
        output_path=args.output,
        term_col=args.term_col,
        font=args.font,
        sigma=args.sigma,
        bg=args.bg,
        min_saturation=args.min_saturation,
        min_value=args.min_value,
    )


if __name__ == "__main__":
    main()