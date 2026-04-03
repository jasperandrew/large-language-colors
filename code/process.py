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
    "avocado": ["avocado green"],
    "blue-violet": ["blue violet","blueviolet"],
    "blush": ["blush pink"],
    "chocolate": ["chocolate brown"],
    "coral": ["coral pink"],
    "dark blue-gray": ["dark blue gray"],
    "dark magenta": ["darkmagenta"],
    "dark olive": ["dark olive green"],
    "dark slate": ["darkslategray"],
    "dark violet": ["darkviolet"],
    "emerald": ["emerald green"],
    "gold": ["golden"],
    "green-yellow": ["greenyellow"],
    "honeydew": ["honeydydew"], # ???
    "jade": ["jade green"],
    "lavender-gray": ["lavender gray"],
    "light coral": ["lightcoral"],
    "light green-gray": ["light greenish-gray"],
    "light olive": ["light olive green"],
    "lime": ["lime green","limegreen"],
    "medium purple": ["mediumpurple"],
    "medium sea green": ["mediumseagreen"],
    "medium spring green": ["mediumspringgreen"],
    "mint": ["mint green"],
    "mustard": ["mustard yellow"],
    "navy": ["navy blue"],
    "olive": ["olive green"],
    "orange-red": ["orange red","orangered"],
    "orchid": ["orchid pink"],
    "pale olive": ["pale olive green"],
    "pistachio": ["pistachio green"],
    "red-brown": ["reddish brown"],
    "rose": ["rose pink"],
    "sage": ["sage green"],
    "salmon": ["salmon pink"],
    "sea green": ["seagreen"],
    "slate": ["slate gray"],
    "teal-blue": ["teal blue"],
    "teal-green": ["teal green"],
    "terracotta": ["terra cotta"],
    "tomato": ["tomato red"],
    "violet-blue": ["violet blue"],
    "wine red": ["wine"],
    "yellow-green": ["yellowgreen"],
}

MAPPING_ZH = {
    "勃艮第色": ["勃艮第红色"],
    "薰衣草色": ["薰衣草紫色"],
    "鲑鱼色": ["鲑鱼粉色"],
    "青柠色": ["青柠绿色"],
    "靛色": ["靛蓝色"],
    "铁锈色": ["铁锈红色", "锈红色", "锈色"],
    "丁香色": ["丁香紫色"],
    "玫瑰色": ["玫瑰红色", "玫红色"], # 枚红色 typo?
    "荧光绿色": ["荧绿色"],
    "碧色": ["碧绿色"],
    "赭色": ["赭石色"],
    # verify
    "柠檬绿色": ["柠绿色"],
    "橘红色": ["桔红色"],
    # light vs. pale
    # "浅粉色": ["淡粉色"],
    # "浅紫色": ["淡紫色"],
    # "浅黄色": ["淡黄色"],
    # "浅绿色": ["淡绿色"],
    # "浅蓝色": ["淡蓝色"],
}

def normalize_term(lang, term):
    if not term: raise ValueError("Null/empty color term")
    term = str(term).strip().lower()
    term = re.sub(r'\s+', ' ', term).strip()
    if lang == "en":
        term = re.sub('grey', 'gray', term)
        for canonical,variants in MAPPING_EN.items():
            if term in variants: return canonical
    if lang == "zh":
        if not term.endswith("色"): term += "色"
    return term

def reduce_term(lang, norm):
    if not norm: raise ValueError("Null/empty color term")
    if lang == "en":
        norm = normalize_term(lang, re.split(r'[-\s]+', norm)[-1])
        return re.split(r'[-\s]+', norm)[-1]
    if lang == "zh":
        norm = normalize_term(lang, list(norm)[-2] + "色")
        return list(norm)[-2] + "色"
    return norm

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} input.csv output.csv", file=sys.stderr)
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
        fields += [f"{c}_raw", f"{c}_norm"] #, f"{c}_min"]

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
                # row[f"{c}_min"] = reduce_term(lang, norm)
            writer.writerow(row)

    print(f"Wrote processed CSV: {outfile}")

if __name__ == "__main__":
    main()
