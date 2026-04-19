import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

# CLI args 
if len(sys.argv) != 3:
    print("Usage: python analyze.py <llm_csv> <mlmc_csv>")
    sys.exit(1)

llm_path  = Path(sys.argv[1])
mlmc_path = Path(sys.argv[2])

# Load 
llm_df  = pd.read_csv(llm_path)
mlmc_df = pd.read_csv(mlmc_path)

sources = {
    "gpt": llm_df["gpt_min"].dropna().value_counts(),
    "gemini":  llm_df["gemini_min"].dropna().value_counts(),
    "human":   mlmc_df["human_min"].dropna().value_counts(),
}

# Align 
def align_distributions(dists):
    df = pd.DataFrame(dists).fillna(0)
    return df / df.sum()

aligned = align_distributions(sources)
raw_counts = {
    s: sources[s].reindex(aligned.index).fillna(0).values.astype(int)
    for s in sources
}

# Metrics 
def jsd(p, q):
    return jensenshannon(p, q, base=2) ** 2

def tvd(p, q):
    return 0.5 * np.abs(p - q).sum()

def spearman(p, q):
    return spearmanr(p, q)

def bootstrap_metrics(counts1, counts2, n_boot=1000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    jsd_boot, tvd_boot, rho_boot = [], [], []
    for _ in range(n_boot):
        s1 = rng.multinomial(counts1.sum(), counts1 / counts1.sum())
        s2 = rng.multinomial(counts2.sum(), counts2 / counts2.sum())
        p, q = s1 / s1.sum(), s2 / s2.sum()
        jsd_boot.append(jsd(p, q))
        tvd_boot.append(tvd(p, q))
        rho_boot.append(spearmanr(p, q)[0])
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {
        "JSD_CI":      (np.percentile(jsd_boot, lo), np.percentile(jsd_boot, hi)),
        "TVD_CI":      (np.percentile(tvd_boot, lo), np.percentile(tvd_boot, hi)),
        "Spearman_CI": (np.percentile(rho_boot, lo), np.percentile(rho_boot, hi)),
    }

# Compare all pairs 
source_names = list(aligned.columns)
results = []
for i, s1 in enumerate(source_names):
    for s2 in source_names[i+1:]:
        p  = aligned[s1].values
        q  = aligned[s2].values
        c1 = raw_counts[s1]
        c2 = raw_counts[s2]
        cis = bootstrap_metrics(c1, c2)
        rho, pval = spearman(p, q)
        results.append({
            "pair":          f"{s1} vs {s2}",
            "JSD":           round(jsd(p, q), 4),
            "JSD_CI":        tuple(round(x, 4) for x in cis["JSD_CI"]),
            "TVD":           round(tvd(p, q), 4),
            "TVD_CI":        tuple(round(x, 4) for x in cis["TVD_CI"]),
            "Spearman_rho":  round(rho, 4),
            "Spearman_pval": round(pval, 4),
            "Spearman_CI":   tuple(round(x, 4) for x in cis["Spearman_CI"]),
        })

# Output 
lang = llm_df["lang_code"].iloc[0] if "lang_code" in llm_df.columns else llm_path.stem
print(f"\n=== Language: {lang} ===\n")

col_w = 22
def fmt_ci(ci):
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"

header = f"{'Pair':<20} {'JSD':>8} {'JSD 95% CI':>16} {'TVD':>8} {'TVD 95% CI':>16} {'Spearman ρ':>11} {'p-value':>9} {'Spearman 95% CI':>18}"
print(header)
print("-" * len(header))
for r in results:
    print(
        f"{r['pair']:<20}"
        f" {r['JSD']:>8.4f}"
        f" {fmt_ci(r['JSD_CI']):>16}"
        f" {r['TVD']:>8.4f}"
        f" {fmt_ci(r['TVD_CI']):>16}"
        f" {r['Spearman_rho']:>11.4f}"
        f" {r['Spearman_pval']:>9.4f}"
        f" {fmt_ci(r['Spearman_CI']):>18}"
    )
print()