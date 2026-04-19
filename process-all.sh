#!/usr/bin/env bash

LANGUAGES=("en" "ko" "zh")

for lang in "${LANGUAGES[@]}"; do
    echo "=== Processing: $lang ==="

    python3 code/processing/extract-mlmc.py "data/mlmc/cleaned_color_names-${lang}.csv" "data/mlmc/${lang}-full-mlmc.csv" --rgbset full
    python3 code/processing/extract-mlmc.py "data/mlmc/cleaned_color_names-${lang}.csv" "data/mlmc/${lang}-hue-mlmc.csv"  --rgbset line

    python3 code/processing/normalize.py "data/raw/${lang}-full-raw.csv" "data/processed/${lang}-full.csv"
    python3 code/processing/normalize.py "data/raw/${lang}-hue-raw.csv"  "data/processed/${lang}-hue.csv"

    echo ""
done

echo "Done."