import colorsys
import numpy as np

def hue_to_rgb(h_idx, n_bins, s=1.0, v=1.0):
    h = (h_idx / n_bins)  # in [0,1)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))

def hue_line(n, n_bins=3600):
    hue_vals = np.random.randint(0, n_bins, size=(n), dtype=np.uint16)
    return list(map(lambda h: hue_to_rgb(h, n_bins), hue_vals))

def full_cube(n):
    return list(map(tuple, np.random.randint(0, 256, size=(n,3), dtype=np.uint8)))

def test_cube(n=4):
    vals = []
    d = 256/n
    mid = 256/n/2
    for ir in range(n):
        for ig in range(n):
            for ib in range(n):
                vals.append((int(d*ir+mid), int(d*ig+mid), int(d*ib+mid)))
    return vals