import numpy as np
import pandas as pd

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

