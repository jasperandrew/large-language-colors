import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
import unicodedata

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument("--en", nargs=2, metavar=("LLM_CSV", "MLMC_CSV"))
parser.add_argument("--ko", nargs=2, metavar=("LLM_CSV", "MLMC_CSV"))
parser.add_argument("--zh", nargs=2, metavar=("LLM_CSV", "MLMC_CSV"))
parser.add_argument("--unweighted", action="store_true", help="Use unweighted average across languages")
args = parser.parse_args()

lang_inputs = {k: v for k, v in vars(args).items() if v is not None and k != "unweighted"}
if not lang_inputs:
    parser.error("Provide at least one language, e.g. --en llm.csv mlmc.csv")

# --- Load ---
def normalize_terms(series):
    return series.dropna().apply(lambda x: unicodedata.normalize("NFC", x))

def load_sources(llm_path, mlmc_path):
    llm_df  = pd.read_csv(llm_path)
    mlmc_df = pd.read_csv(mlmc_path)
    return {
        "gpt":    normalize_terms(llm_df["gpt_min"]).value_counts(),
        "gemini": normalize_terms(llm_df["gemini_min"]).value_counts(),
        "human":  normalize_terms(mlmc_df["human_min"]).value_counts(),
    }

# --- Align ---
def align_distributions(dists):
    df = pd.DataFrame(dists).fillna(0)
    return df / df.sum()

# --- Metrics ---
def jsd(p, q):
    return jensenshannon(p, q, base=2) ** 2

def spearman(p, q):
    return spearmanr(p, q)

def bootstrap_metrics(counts1, counts2, n_boot=1000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    jsd_boot, rho_boot = [], []
    for _ in range(n_boot):
        s1 = rng.multinomial(counts1.sum(), counts1 / counts1.sum())
        s2 = rng.multinomial(counts2.sum(), counts2 / counts2.sum())
        p, q = s1 / s1.sum(), s2 / s2.sum()
        jsd_boot.append(jsd(p, q))
        rho_boot.append(spearmanr(p, q)[0])
    
    jsd_arr = np.array(jsd_boot)
    rho_arr = np.array(rho_boot)
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {
        "JSD":      np.mean(jsd_arr),
        "JSD_boot": jsd_arr,
        "JSD_CI":   (np.percentile(jsd_arr, lo), np.percentile(jsd_arr, hi)),
        "Rho":      np.mean(rho_arr),
        "Rho_boot": rho_arr,
        "Rho_CI":   (np.percentile(rho_arr, lo), np.percentile(rho_arr, hi)),
    }

def harmonic_n(n1, n2):
    return 2 * n1 * n2 / (n1 + n2)

def split_half_reliability(counts, n_splits=100, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    jsd_boot = []
    terms = np.repeat(np.arange(len(counts)), counts)
    for _ in range(n_splits):
        rng.shuffle(terms)
        mid = len(terms) // 2
        half1, half2 = terms[:mid], terms[mid:]
        p = np.bincount(half1, minlength=len(counts)) / len(half1)
        q = np.bincount(half2, minlength=len(counts)) / len(half2)
        jsd_boot.append(jsd(p, q))
    jsd_arr = np.array(jsd_boot)
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {
        "JSD":      np.mean(jsd_arr),
        "JSD_CI":   (np.percentile(jsd_arr, lo), np.percentile(jsd_arr, hi)),
        "JSD_boot": jsd_arr,  # keep this for weighted average
    }

# --- Per-language computation ---
def compute_lang(sources):
    aligned = align_distributions(sources)
    raw_counts = {
        s: sources[s].reindex(aligned.index).fillna(0).values.astype(int)
        for s in sources
    }

    human_counts = raw_counts["human"]
    sh = split_half_reliability(human_counts)
    results = {
        "human split-half": {
            "JSD":      round(sh["JSD"], 4),
            "JSD_CI":   tuple(round(x, 4) for x in sh["JSD_CI"]),
            "JSD_boot": sh["JSD_boot"],  # now populated
            "Rho":      None,
            "Rho_CI":   None,
            "Rho_boot": None,
            "n1":       human_counts.sum() // 2,
            "n2":       human_counts.sum() // 2,
        }
    }

    source_names = list(aligned.columns)
    for i, s1 in enumerate(source_names):
        for s2 in source_names[i+1:]:
            p  = aligned[s1].values
            q  = aligned[s2].values
            c1 = raw_counts[s1]
            c2 = raw_counts[s2]
            boot = bootstrap_metrics(c1, c2)
            results[f"{s1} vs {s2}"] = {
                "JSD":      round(boot["JSD"], 4),
                "JSD_CI":   tuple(round(x, 4) for x in boot["JSD_CI"]),
                "JSD_boot": boot["JSD_boot"],
                "Rho":      round(boot["Rho"], 4),
                "Rho_CI":   tuple(round(x, 4) for x in boot["Rho_CI"]),
                "Rho_boot": boot["Rho_boot"],
                "n1":       int(c1.sum()),
                "n2":       int(c2.sum()),
            }

    return results

# --- Weighted average across languages ---
def weighted_summarize(boot_arrays, ns, point_estimates, ci=95):
    weights = np.array(ns, dtype=float) / sum(ns)
    weighted_mean = sum(e * w for e, w in zip(point_estimates, weights))
    boot_means = np.zeros(len(boot_arrays[0]))
    for arr, w in zip(boot_arrays, weights):
        boot_means += arr * w
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return weighted_mean, np.percentile(boot_means, lo), np.percentile(boot_means, hi)

# --- Output helpers ---
def fmt_ci(ci):
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"

def print_table(rows):
    header = f"{'Pair':<20} {'JSD':>8} {'JSD 95% CI':>16} {'Spearman ρ':>11} {'Spearman 95% CI':>18}"
    print(header)
    print("-" * len(header))
    for pair, r in rows:
        rho_str = f"{r['Rho']:>11.4f}" if r['Rho'] is not None else f"{'—':>11}"
        ci_str  = fmt_ci(r['Rho_CI']) if r['Rho_CI'] is not None else f"{'—':>18}"
        print(
            f"{pair:<20}"
            f" {r['JSD']:>8.4f}"
            f" {fmt_ci(r['JSD_CI']):>16}"
            f" {rho_str}"
            f" {ci_str:>18}"
        )
    print()

# --- Run ---
all_results = {}
for lang, (llm_path, mlmc_path) in lang_inputs.items():
    sources = load_sources(llm_path, mlmc_path)
    all_results[lang] = compute_lang(sources)
    print(f"\n=== Language: {lang} ===\n")
    print_table(all_results[lang].items())

# --- Cross-language weighted averages ---
pair_types = {
    "human split-half": "Human split-half",
    "gpt vs gemini":    "LLM vs LLM",
    "gpt vs human":     "GPT vs Human",
    "gemini vs human":  "Gemini vs Human",
}

print(f"\n=== {'Unweighted' if args.unweighted else 'Weighted'} Average Across Languages ===\n")
header = f"{'Pair':<20} {'JSD':>8} {'JSD 95% CI':>16} {'Spearman ρ':>11} {'Spearman 95% CI':>18}"
print(header)
print("-" * len(header))

for pair, label in pair_types.items():
    lang_data = [all_results[lang][pair] for lang in all_results if pair in all_results[lang]]
    if not lang_data:
        continue

    if args.unweighted:
        ns = [1] * len(lang_data)
    else:
        ns = [harmonic_n(d["n1"], d["n2"]) for d in lang_data]

    jsd_mean, jsd_lo, jsd_hi = weighted_summarize(
        [d["JSD_boot"] for d in lang_data], ns, [d["JSD"] for d in lang_data])

    # Skip rho if any language has None (i.e. split-half)
    if any(d["Rho_boot"] is None for d in lang_data):
        rho_str = f"{'—':>11}"
        ci_str  = f"{'—':>18}"
    else:
        rho_mean, rho_lo, rho_hi = weighted_summarize(
            [d["Rho_boot"] for d in lang_data], ns, [d["Rho"] for d in lang_data])
        rho_str = f"{rho_mean:>11.4f}"
        ci_str  = fmt_ci((rho_lo, rho_hi))

    print(
        f"{label:<20}"
        f" {jsd_mean:>8.4f}"
        f" {fmt_ci((jsd_lo, jsd_hi)):>16}"
        f" {rho_str}"
        f" {ci_str:>18}"
    )

print()