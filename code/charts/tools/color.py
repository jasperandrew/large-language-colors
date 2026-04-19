import numpy as np
import pandas as pd


def _rgb_norm_to_sat_val(r, g, b):
    cmax = np.maximum.reduce([r, g, b])
    cmin = np.minimum.reduce([r, g, b])
    delta = cmax - cmin
    sat = np.zeros(len(r))
    nonzero = cmax > 0
    sat[nonzero] = delta[nonzero] / cmax[nonzero]
    return sat, cmax  # cmax == value


def rgb_to_sat_val(r, g, b):
    """Return HSV saturation and value arrays from RGB (0–255)."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    return _rgb_norm_to_sat_val(r,g,b)


def rgb_to_lab(r, g, b):
    """Vectorized sRGB (0–255) → CIELAB (D65)."""
    rgb = np.stack([r, g, b], axis=1) / 255.0
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
    eps, kap = 0.008856, 903.3
    fx = np.where(xyz > eps, xyz ** (1 / 3), (kap * xyz + 16) / 116)
    L  = 116 * fx[:, 1] - 16
    a  = 500 * (fx[:, 0] - fx[:, 1])
    b_ = 200 * (fx[:, 1] - fx[:, 2])
    return L, a, b_


def df_hues(df: pd.DataFrame) -> np.ndarray:
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


def df_sat_val(df: pd.DataFrame):
    """Return (saturation, value) arrays for HSV, vectorized, from 0–255 RGB."""
    r = df["r"].to_numpy(dtype=float) / 255
    g = df["g"].to_numpy(dtype=float) / 255
    b = df["b"].to_numpy(dtype=float) / 255
    return _rgb_norm_to_sat_val(r,g,b)


def df_term_colors(df: pd.DataFrame, term_col: str) -> dict[str, tuple]:
    """
    Mean RGB per color term as (r, g, b) in [0, 1].
    Vectorized via groupby — much faster than row-wise apply.
    """
    means = df.groupby(term_col)[["r", "g", "b"]].mean() / 255
    return {term: (row["r"], row["g"], row["b"]) for term, row in means.iterrows()}
