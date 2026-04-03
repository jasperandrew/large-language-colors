import argparse
import colorsys
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
from matplotlib.colors import to_hex


def rgb_to_hue(r, g, b):
    """Return hue in degrees [0, 360) for an RGB triple (0–255 each)."""
    h, _, _ = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return h * 360


def average_rgb(group):
    """Compute the mean RGB of a DataFrame subset and return as (r,g,b) 0–1."""
    r = group["r"].mean() / 255
    g = group["g"].mean() / 255
    b = group["b"].mean() / 255
    return (r, g, b)


def set_font(font_spec):
    """
    Configure matplotlib to use a font.
    font_spec can be:
      - None: do nothing
      - path to a font file: registers that font and uses it
      - font family name (string): uses that family name
    """
    if not font_spec:
        return

    try:
        # If it's a file path, register the font and use its internal name
        if os.path.exists(font_spec):
            font_manager.fontManager.addfont(font_spec)
            fp = font_manager.FontProperties(fname=font_spec)
            name = fp.get_name()
            mpl.rcParams["font.family"] = "sans-serif"
            # put the registered name first so it's preferred
            mpl.rcParams["font.sans-serif"] = [name] + mpl.rcParams.get(
                "font.sans-serif", []
            )
        else:
            # Treat as a font family name
            mpl.rcParams["font.family"] = font_spec
    except Exception as e:
        print(f"Warning: failed to set font '{font_spec}': {e}")

    # Ensure minus sign renders correctly
    mpl.rcParams["axes.unicode_minus"] = False


def build_chart(
    csv_path: str,
    bin_width: float = 5.0,
    output_path: str = None,
    term_col: str = "color_term",
    top_n: int = None,
    font: str = None,
):
    # Configure font early so axis/legend/title use it
    set_font(font)

    # ── Load data ────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    required = {"r", "g", "b", term_col}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    df["hue"] = df.apply(lambda row: rgb_to_hue(row["r"], row["g"], row["b"]), axis=1)

    # ── Bin hues ─────────────────────────────────────────────────────────────
    bins = np.arange(0, 360 + bin_width, bin_width)
    bin_labels = bins[:-1]  # left edge of each bin
    df["hue_bin"] = pd.cut(df["hue"], bins=bins, labels=bin_labels, right=False)
    df["hue_bin"] = df["hue_bin"].astype(float)

    # ── Compute per-(bin, term) counts and proportions ────────────────────────
    counts = (
        df.groupby(["hue_bin", term_col], observed=True)
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("hue_bin", observed=True)["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals

    # ── Compute average color per color term (across ALL data points) ─────────
    term_colors = (
        df.groupby(term_col)
        .apply(average_rgb, include_groups=False)
        .to_dict()
    )

    # ── Optionally restrict to top N terms ──────────────────────────────────
    if top_n is not None and top_n > 0:
        top_terms = df[term_col].value_counts().nlargest(top_n).index.tolist()
        # Filter counts to only these terms and recompute proportions among them
        counts = counts[counts[term_col].isin(top_terms)].copy()
        totals = counts.groupby("hue_bin", observed=True)["count"].transform("sum")
        counts["proportion"] = counts["count"] / totals
        counts["proportion"] = counts["proportion"].fillna(0)

        # Keep only colors for the selected terms
        term_colors = {k: v for k, v in term_colors.items() if k in top_terms}

    # ── Pivot to wide format for stacking ────────────────────────────────────
    pivot = counts.pivot_table(
        index="hue_bin", columns=term_col, values="proportion", aggfunc="sum"
    ).fillna(0)

    # Ensure all bins are present in the index (so x-axis covers full 0..360)
    pivot = pivot.reindex(bin_labels, fill_value=0)

    # Sort bins numerically
    pivot = pivot.sort_index()

    color_terms = pivot.columns.tolist()
    x = pivot.index.values

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(12, len(x) * 0.12), 6))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    bottom = np.zeros(len(x))
    for term in color_terms:
        heights = pivot[term].values
        color = term_colors.get(term, (0.5, 0.5, 0.5))
        ax.bar(
            x,
            heights,
            width=bin_width,
            bottom=bottom,
            color=color,
            align="edge",
            linewidth=0,
        )
        bottom += heights

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(
            color=term_colors.get(t, (0.5, 0.5, 0.5)),
            label=t,
        )
        for t in color_terms
    ]
    legend = ax.legend(
        handles=patches,
        loc="upper right",
        bbox_to_anchor=(1.18, 1),
        framealpha=0.15,
        facecolor="#333333",
        edgecolor="none",
        labelcolor="white",
        fontsize=8,
        title="Color term",
        title_fontsize=9,
    )
    legend.get_title().set_color("white")

    # ── Axes styling ──────────────────────────────────────────────────────────
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Hue (degrees)", color="white", fontsize=11)
    ax.set_ylabel("Proportion", color="white", fontsize=11)
    ax.set_title(
        f"Color term distribution by hue  (bin width = {bin_width}°)",
        color="white",
        fontsize=13,
        pad=14,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

    # Add hue reference ticks
    ax.set_xticks(np.arange(0, 361, 30))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"Saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Stacked hue chart from RGB+color_term CSV.")
    parser.add_argument("csv", help="Path to CSV file with columns: r, g, b, color_term")
    parser.add_argument(
        "--bin-width",
        type=float,
        default=5.0,
        help="Width of each hue bin in degrees (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the chart (e.g. chart.png). If omitted, displays interactively.",
    )
    parser.add_argument(
        "--term-col",
        type=str,
        default="color_term",
        help="Name of CSV column containing color terms (default: 'color_term')",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only include the top N most frequent color terms (by overall count). If omitted, include all terms.",
    )
    parser.add_argument(
        "--font",
        type=str,
        default=None,
        help="Font family name or path to a font file to use for rendering text (e.g. 'SimHei' or '/path/to/NotoSansCJKsc-Regular.otf').",
    )
    args = parser.parse_args()
    build_chart(
        args.csv,
        bin_width=args.bin_width,
        output_path=args.output,
        term_col=args.term_col,
        top_n=args.top_n,
        font=args.font,
    )


if __name__ == "__main__":
    main()
