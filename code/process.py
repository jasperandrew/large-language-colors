#!/usr/bin/env python3
"""
clean_colors_onefile.py

Usage:
  python clean_colors_onefile.py input.csv output.csv

Hard-coded mapping at the top. No other options.
"""

import sys
import csv
import re
from collections import defaultdict, Counter

MAPPING_EN = {
    "lime": ["lime green","limegreen"],
    "olive": ["olive green"],
    "sage": ["sage green"],
    "yellow-green": ["yellowgreen"],
    "green-yellow": ["greenyellow"],
    "avocado": ["avocado green"],
    "blue-violet": ["blue violet","blueviolet"],
    "blush": ["blush pink"],
    "chocolate": ["chocolate brown"],
    "coral": ["coral pink"],
    "dark blue-gray": ["dark blue gray"],
    "dark olive": ["dark olive green"],
    "dark magenta": ["darkmagenta"],
    "dark slate": ["darkslategray"],
    "dark violet": ["darkviolet"],
    "emerald": ["emerald green"],
    "gold": ["golden"],
    "honeydew": ["honeydydew"], # ???
    "jade": ["jade green"],
    "lavender-gray": ["lavender gray"],
    "light green-gray": ["light greenish-gray"],
    "light olive": ["light olive green"],
    "light coral": ["lightcoral"],
    "medium purple": ["mediumpurple"],
    "medium sea green": ["mediumseagreen"],
    "mint": ["mint green"],
    "mustard": ["mustard yellow"],
    "navy": ["navy blue"],
    "orange-red": ["orange red","orangered"],
    "orchid": ["orchid pink"],
    "pale olive": ["pale olive green"],
    "pistachio": ["pistachio green"],
    "red-brown": ["reddish brown"],
    "rose": ["rose pink"],
    "salmon": ["salmon pink"],
    "sea green": ["seagreen"],
    "slate": ["slate gray"],
    "teal-blue": ["teal blue"],
    "teal-green": ["teal green"],
    "terracotta": ["terra cotta"],
    "tomato": ["tomato red"],
    "violet-blue": ["violet blue"],
    "wine red": ["wine"],
}

def normalize_term(lang, term):
    if not term: raise ValueError("Null color term")
    term = str(term).strip().lower()
    term = re.sub(r'\s+', ' ', term).strip()
    if lang == "en":
        term = re.sub('grey', 'gray', term)
        for canonical,variants in MAPPING_EN.items():
            if term in variants: return canonical
    if lang == "zh": pass
    return term

def reduce_term(lang, norm):
    if not norm: raise ValueError("Null color term")
    if lang == "en":
        norm = normalize_term(lang, re.split(r'[-\s]+', norm)[-1])
        return re.split(r'[-\s]+', norm)[-1]
    if lang == "zh": pass
    return norm

def main():
    if len(sys.argv) != 3:
        print("Usage: python clean_colors_onefile.py input.csv output.csv", file=sys.stderr)
        sys.exit(2)
    infile = sys.argv[1]
    outfile = sys.argv[2]

    with open(infile, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    color_cols = [c for c in ("chatgpt", "gemini") if c in fieldnames]
    if not color_cols:
        print("Input must contain 'chatgpt' and/or 'gemini' columns.", file=sys.stderr)
        sys.exit(1)

    fields = list(fieldnames)[:4]
    for c in color_cols:
        fields += [f"{c}_raw", f"{c}_norm", f"{c}_min"]

    # for r in rows:
    #     for c in color_cols:
    #         lang = r.get("lang_code")
    #         raw = r.get(c, "")
    #         norm = normalize_term(lang, raw)
    #         r[f"{c}_norm"] = norm
    #         r[f"{c}_min"] = reduce_term(lang, norm)

    with open(outfile, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in fields[:4]}
            for c in color_cols:
                lang = r.get("lang_code")
                raw = r.get(c, "")
                norm = normalize_term(lang, raw)
                row[f"{c}_raw"] = raw
                row[f"{c}_norm"] = norm
                row[f"{c}_min"] = reduce_term(lang, norm)
            writer.writerow(row)

    print(f"Wrote processed CSV: {outfile}")

if __name__ == "__main__":
    main()
