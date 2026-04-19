#!/usr/bin/env bash

LANGUAGES=("en" "ko" "zh")

TOP_N=15
N_BINS=120
TEXT_COLOR=white

for lang in "${LANGUAGES[@]}"; do
    echo "=== Charting: $lang ==="

    python3 code/stats/chart.py "data/mlmc/${lang}-hue-mlmc.csv" --term-col human_min   --top-n $TOP_N --n-bins $N_BINS --output charts/${lang}-human.png   --text-color $TEXT_COLOR
    python3 code/stats/chart.py "data/processed/${lang}-hue.csv" --term-col chatgpt_min --top-n $TOP_N --n-bins $N_BINS --output charts/${lang}-chatgpt.png --text-color $TEXT_COLOR
    python3 code/stats/chart.py "data/processed/${lang}-hue.csv" --term-col gemini_min  --top-n $TOP_N --n-bins $N_BINS --output charts/${lang}-gemini.png  --text-color $TEXT_COLOR

    echo ""
done

echo "Done."