import sys

def print_usage():
    print(f"Usage: python {sys.argv[0]} <CSV_PATH=str>")

if len(sys.argv) <= 1:
    print_usage()
    exit(1)

csv_path = sys.argv[1]

import csv
import numpy as np
from scipy.stats import chisquare

rgb_data = []
with open(csv_path,'r') as f:
    reader = csv.reader(f)

    for i, line in enumerate(reader):
        if i == 0: continue
        r,g,b = line[:3]
        rgb_data.append([int(r),int(g),int(b)])

rgb_data = np.array([np.array(xi) for xi in rgb_data])

print("mean: ", rgb_data.mean(axis=0))
print("std:  ", rgb_data.std(axis=0))

for i, ch in enumerate("RGB"):
    counts, _ = np.histogram(rgb_data[:, i], bins=256, range=(0, 256))
    stat, p = chisquare(counts)
    print(f"{ch}: chi2={stat:.2f}, p={p:.4f}") # p >> 0.05 means uniform

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for i, (ax, ch, color) in enumerate(zip(axes, "RGB", ["red", "green", "blue"])):
    ax.hist(rgb_data[:, i], bins=64, color=color, alpha=0.7)
    ax.set_title(f"{ch} channel")
    ax.set_xlim(0, 255)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
pairs = [("R", "G", 0, 1), ("R", "B", 0, 2), ("G", "B", 1, 2)]

for ax, (a, b, i, j) in zip(axes, pairs):
    ax.hist2d(rgb_data[:, i], rgb_data[:, j], bins=64, cmap="viridis")
    ax.set_xlabel(a); ax.set_ylabel(b)
    ax.set_title(f"{a} vs {b}")
plt.tight_layout()
plt.show()