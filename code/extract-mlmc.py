import csv
import argparse
import sys
import re

def main():
    p = argparse.ArgumentParser(description="Extract columns [lang,name,r,g,b] and optionally filter by lang and rgbSet.")
    p.add_argument("input_csv", help="Input CSV file path")
    p.add_argument("output_csv", help="Output CSV file path")
    p.add_argument("--rgbset", help="rgbSet value to keep. Exact match against the 'rgbSet' column.")
    args = p.parse_args()

    keep_cols = ['langAbv', 'name', 'r', 'g', 'b']

    with open(args.input_csv, newline='', encoding='utf-8') as inf, \
         open(args.output_csv, 'w', newline='', encoding='utf-8') as outf:

        reader = csv.DictReader(inf)
        # check required columns exist
        for col in keep_cols + ['rgbSet']:
            if col not in reader.fieldnames:
                sys.exit(f"ERROR: input CSV missing required column: {col}")

        writer = csv.DictWriter(outf, fieldnames=['r','g','b','lang_code','human'])
        writer.writeheader()

        for row in reader:
            if args.rgbset is not None and row['rgbSet'] != args.rgbset: continue
            if row["langAbv"] == "en":
                clean = re.sub(r'[^a-z\-\'\s]', "", re.sub(r'\s', " ", row["name"].lower().strip())).strip()
                if not clean: continue

            try:
                out_row = {c: int(row[c].strip()) for c in ["r","g","b"]}
            except:
                continue

            out_row["lang_code"] = row["langAbv"]
            out_row["human"] = row["name"]

            writer.writerow(out_row)

if __name__ == '__main__':
    main()
