#!/usr/bin/env bash

LANGUAGES=("en" "ko" "zh")

TOP_N=15
N_BINS=120
OUT_FOLDER=chart-output
FONT="Noto Sans CJK JP"

for lang in "${LANGUAGES[@]}"; do
    echo "=== Charting: $lang ==="

    python3 code/charts/hue_stack.py  "data/mlmc/${lang}-hue-mlmc.csv" --term-col human_min   --top-n $TOP_N --n-bins $N_BINS --output $OUT_FOLDER/stack-${lang}-human.png    --font "$FONT"
    python3 code/charts/hue_stack.py  "data/processed/${lang}-hue.csv" --term-col chatgpt_min --top-n $TOP_N --n-bins $N_BINS --output $OUT_FOLDER/stack-${lang}-chatgpt.png  --font "$FONT"
    python3 code/charts/hue_stack.py  "data/processed/${lang}-hue.csv" --term-col gemini_min  --top-n $TOP_N --n-bins $N_BINS --output $OUT_FOLDER/stack-${lang}-gemini.png   --font "$FONT"

    echo ""
    
    python3 code/charts/hue_unique.py "data/mlmc/${lang}-hue-mlmc.csv" --term-col human_min   --sigma 1      --n-bins $N_BINS --output $OUT_FOLDER/unique-${lang}-human.png   --font "$FONT"
    python3 code/charts/hue_unique.py "data/processed/${lang}-hue.csv" --term-col chatgpt_min --sigma 1      --n-bins $N_BINS --output $OUT_FOLDER/unique-${lang}-chatgpt.png --font "$FONT"
    python3 code/charts/hue_unique.py "data/processed/${lang}-hue.csv" --term-col gemini_min  --sigma 1      --n-bins $N_BINS --output $OUT_FOLDER/unique-${lang}-gemini.png  --font "$FONT"

    echo ""
done

echo "Done."