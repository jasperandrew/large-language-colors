import csv
import numpy as np
from scipy.stats import chisquare

samples = []
with open('results.csv','r') as f:
    reader = csv.reader(f)

    for i, line in enumerate(reader):
        if i == 0: continue
        r,g,b = line[:3]
        samples.append([int(r),int(g),int(b)])

samples = np.array([np.array(xi) for xi in samples])

print("mean: ", samples.mean(axis=0))   # per channel
print("std:  ", samples.std(axis=0))    # per channel

for i, ch in enumerate("RGB"):
    counts, _ = np.histogram(samples[:, i], bins=256, range=(0, 256))
    stat, p = chisquare(counts)
    print(f"{ch}: chi2={stat:.2f}, p={p:.4f}")  # p >> 0.05 means uniform

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for i, (ax, ch, color) in enumerate(zip(axes, "RGB", ["red", "green", "blue"])):
    ax.hist(samples[:, i], bins=64, color=color, alpha=0.7)
    ax.set_title(f"{ch} channel")
    ax.set_xlim(0, 255)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
pairs = [("R", "G", 0, 1), ("R", "B", 0, 2), ("G", "B", 1, 2)]

for ax, (a, b, i, j) in zip(axes, pairs):
    ax.hist2d(samples[:, i], samples[:, j], bins=64, cmap="viridis")
    ax.set_xlabel(a); ax.set_ylabel(b)
    ax.set_title(f"{a} vs {b}")
plt.tight_layout()
plt.show()