import argparse
import csv
import sys
from collections import Counter


def list_unique_color_terms(csv_path, sort_alpha=False, only_col=None, min_freq=0):
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        cols = [only_col] if only_col else reader.fieldnames[4:]
        counters = {col: Counter() for col in cols}
        header_map = {name.lower(): name for name in reader.fieldnames}
        actual_cols = {}
        missing = []
        for col in cols:
            if col in header_map:
                actual_cols[col] = header_map[col]
            else:
                if col in reader.fieldnames:
                    actual_cols[col] = col
                else:
                    missing.append(col)
        if missing:
            print(f"Warning: missing columns in CSV header: {missing}", file=sys.stderr)
        for row in reader:
            for col in cols:
                if col not in actual_cols:
                    continue
                raw = row.get(actual_cols[col], "")
                key = raw.lower()
                counters[col][key] += 1

    for col in cols:
        cnt = counters[col]
        total_cells = sum(cnt.values())
        distinct = len(cnt)
        print(f"\nUnique terms in column '{col}': total terms found = {total_cells}, distinct = {distinct}")
        if distinct == 0:
            print("  (none)")
            continue
        if sort_alpha:
            for term, freq in sorted(cnt.items(), key=lambda x: x[0]):
                if freq >= min_freq:
                    print(f"  {term}  ({freq})")
        else:
            for term, freq in cnt.most_common():
                if freq >= min_freq:
                    print(f"  {term}  ({freq})")


def main():
    p = argparse.ArgumentParser(description="List unique color terms and frequencies from CSV columns.")
    p.add_argument("csv", help="CSV file path")
    p.add_argument("--column", help="Only count terms in this column (default: all columns after the 4th)")
    p.add_argument("--sort-alpha", action="store_true", help="Sort terms alphabetically instead of by frequency")
    p.add_argument("--min-freq", type=int, default=0, help="Skip terms with frequency < this value (default: 0)")
    args = p.parse_args()

    list_unique_color_terms(args.csv, sort_alpha=args.sort_alpha, only_col=args.column, min_freq=args.min_freq)


if __name__ == "__main__":
    main()