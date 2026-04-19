import sys
import csv
import re
import unicodedata
from collections import defaultdict, Counter
from lang_norm import get_normalizer

def normalize_term(lang_code, term):
    if not term: raise ValueError("Null/empty color term")
    term = unicodedata.normalize("NFC", term)
    term = re.sub(r'\s+', ' ', term.lower()).strip()
    return get_normalizer(lang_code).process(term)

def minify_term(term):
    return re.sub(r'[\s\-\'\(\)\,]', '', term)

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

    color_cols = fieldnames[4:]
    if not color_cols:
        print("Input must contain color term columns.", file=sys.stderr)
        sys.exit(1)

    fields = list(fieldnames)[:4] # [r,g,b,lang_code]
    for c in color_cols:
        fields += [f"{c}_raw", f"{c}_norm", f"{c}_min"]

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
                row[f"{c}_min"] = minify_term(norm)
            writer.writerow(row)

    print(f"Wrote processed CSV: {outfile}")

if __name__ == "__main__":
    main()
