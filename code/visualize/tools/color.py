import numpy as np
import pandas as pd
from skimage.color import rgb2lab


def _vec_rgb_norm(df: pd.DataFrame):
    def norm(x): return df[x].to_numpy(dtype=float) / 255
    return norm("r"), norm("g"), norm("b")


def _rgb_norm_to_sv(r, g, b):
    cmax = np.maximum.reduce([r, g, b])
    cmin = np.minimum.reduce([r, g, b])
    delta = cmax - cmin
    sat = np.zeros(len(r))
    nonzero = cmax > 0
    sat[nonzero] = delta[nonzero] / cmax[nonzero]
    return sat, cmax  # cmax == value


def rgb_to_sv(r, g, b):
    """Return HSV saturation and value arrays from RGB (0–255)."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    return _rgb_norm_to_sv(r,g,b)


def vec_rgb_to_sv(df: pd.DataFrame):
    """Return (saturation, value) arrays for HSV, vectorized, from 0–255 RGB."""
    return _rgb_norm_to_sv(*_vec_rgb_norm(df))


# https://kaizoudou.com/from-rgb-to-lab-color-space/
def vec_rgb_to_lab(df: pd.DataFrame):
    """Vectorized sRGB (0–255) → CIELAB (D65)."""
    r,g,b = _vec_rgb_norm(df)
    rgb = np.stack([r, g, b], axis=1)
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] /= 12.92
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = rgb @ M.T
    xyz /= np.array([0.95047, 1.00000, 1.08883])
    fx = np.where(xyz > 0.008856, xyz ** (1 / 3), 7.787 * xyz + (16 / 116))
    L_  = 116 * fx[:, 1] - 16
    a_  = 500 * (fx[:, 0] - fx[:, 1])
    b_ = 200 * (fx[:, 1] - fx[:, 2])
    return L_, a_, b_


def vec_rgb_to_h(df: pd.DataFrame) -> np.ndarray:
    """
    Vectorized HSV hue for a DataFrame with r, g, b columns (0–255).
    Returns an array of hues in degrees [0, 360).
    """
    r,g,b = _vec_rgb_norm(df)

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


def term_colors(df: pd.DataFrame, term_col: str) -> dict[str, tuple]:
    """
    Mean RGB per color term as (r, g, b) in [0, 1].
    Vectorized via groupby — much faster than row-wise apply.
    """
    means = df.groupby(term_col)[["r", "g", "b"]].mean() / 255
    return {term: (row["r"], row["g"], row["b"]) for term, row in means.iterrows()}


def filter_achromatic(df: pd.DataFrame, min_sat: float, min_val: float):
    """Filter achromatic colors."""
    sat, val = vec_rgb_to_sv(df)
    chromatic = (sat >= min_sat) & (val >= min_val)
    n_dropped = (~chromatic).sum()
    if n_dropped:
        print(f"Filtered {n_dropped} achromatic rows "
              f"(sat < {min_sat} or val < {min_val})")

    ch_df = df[chromatic].reset_index(drop=True)
    if ch_df.empty:
        raise ValueError(
            "No chromatic rows remain after filtering. "
            "Try lowering --min-sat or --min-val."
        )
    
    return ch_df