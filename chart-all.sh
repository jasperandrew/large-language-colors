#!/usr/bin/env bash

LANGUAGES=("en" "ko" "zh")

TOP_N=15
L_BINS=50
FONT="Noto Sans CJK JP"
OUT_FOLDER=chart-output
VMAX=5

N_BINS=$(( $L_BINS*3 ))

for lang in "${LANGUAGES[@]}"; do
    echo "=== Charting: $lang ==="

    python3 code/charts/hue_stack.py  "data/mlmc/${lang}-hue-mlmc.csv"  --term-col human_min   --n-bins $N_BINS --output $OUT_FOLDER/stack-${lang}-human.png     --font "$FONT" --top-n $TOP_N
    python3 code/charts/hue_stack.py  "data/processed/${lang}-hue.csv"  --term-col chatgpt_min --n-bins $N_BINS --output $OUT_FOLDER/stack-${lang}-chatgpt.png   --font "$FONT" --top-n $TOP_N
    python3 code/charts/hue_stack.py  "data/processed/${lang}-hue.csv"  --term-col gemini_min  --n-bins $N_BINS --output $OUT_FOLDER/stack-${lang}-gemini.png    --font "$FONT" --top-n $TOP_N

    echo ""
    
    python3 code/charts/hue_unique.py "data/mlmc/${lang}-hue-mlmc.csv"  --term-col human_min   --n-bins $N_BINS --output $OUT_FOLDER/unique-${lang}-human.png    --font "$FONT"
    python3 code/charts/hue_unique.py "data/processed/${lang}-hue.csv"  --term-col chatgpt_min --n-bins $N_BINS --output $OUT_FOLDER/unique-${lang}-chatgpt.png  --font "$FONT"
    python3 code/charts/hue_unique.py "data/processed/${lang}-hue.csv"  --term-col gemini_min  --n-bins $N_BINS --output $OUT_FOLDER/unique-${lang}-gemini.png   --font "$FONT"

    echo ""

    python3 code/charts/entropy.py    "data/mlmc/${lang}-full-mlmc.csv" --term-col human_min   --l-bins $L_BINS --output $OUT_FOLDER/entropy-${lang}-human.png   --font "$FONT" --vmax $VMAX
    python3 code/charts/entropy.py    "data/processed/${lang}-full.csv" --term-col chatgpt_min --l-bins $L_BINS --output $OUT_FOLDER/entropy-${lang}-chatgpt.png --font "$FONT" --vmax $VMAX
    python3 code/charts/entropy.py    "data/processed/${lang}-full.csv" --term-col gemini_min  --l-bins $L_BINS --output $OUT_FOLDER/entropy-${lang}-gemini.png  --font "$FONT" --vmax $VMAX

    echo ""
done

echo "Done."