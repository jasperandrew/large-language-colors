#!/usr/bin/env bash

LANGUAGES=("en" "ko" "zh")

TOP_N=15
L_BINS=50
FONT="Noto Sans CJK JP"
OUT_FOLDER=chart-output
YMAX=20
VMAX=4

N_BINS=$(( $L_BINS*3 ))

for lang in "${LANGUAGES[@]}"; do
    echo "=== Charting: $lang ==="

    python3 code/visualize/hue_stack.py  "data/mlmc/$lang-hue-mlmc.csv"  $lang Human  human_min   --n-bins $N_BINS --output $OUT_FOLDER/stack-$lang-human.png     --font "$FONT" --top-n $TOP_N
    python3 code/visualize/hue_stack.py  "data/processed/$lang-hue.csv"  $lang GPT    gpt_min     --n-bins $N_BINS --output $OUT_FOLDER/stack-$lang-gpt.png       --font "$FONT" --top-n $TOP_N
    python3 code/visualize/hue_stack.py  "data/processed/$lang-hue.csv"  $lang Gemini gemini_min  --n-bins $N_BINS --output $OUT_FOLDER/stack-$lang-gemini.png    --font "$FONT" --top-n $TOP_N

    echo ""
    
    python3 code/visualize/hue_unique.py "data/mlmc/$lang-hue-mlmc.csv"  $lang Human  human_min   --n-bins $N_BINS --output $OUT_FOLDER/unique-$lang-human.png    --font "$FONT" # --ymax $YMAX
    python3 code/visualize/hue_unique.py "data/processed/$lang-hue.csv"  $lang GPT    gpt_min     --n-bins $N_BINS --output $OUT_FOLDER/unique-$lang-gpt.png      --font "$FONT" # --ymax $YMAX
    python3 code/visualize/hue_unique.py "data/processed/$lang-hue.csv"  $lang Gemini gemini_min  --n-bins $N_BINS --output $OUT_FOLDER/unique-$lang-gemini.png   --font "$FONT" # --ymax $YMAX

    echo ""

    python3 code/visualize/entropy.py    "data/mlmc/$lang-full-mlmc.csv" $lang Human  human_min   --l-bins $L_BINS --output $OUT_FOLDER/entropy-$lang-human.png   --font "$FONT" --vmax $VMAX
    python3 code/visualize/entropy.py    "data/processed/$lang-full.csv" $lang GPT    gpt_min     --l-bins $L_BINS --output $OUT_FOLDER/entropy-$lang-gpt.png     --font "$FONT" --vmax $VMAX
    python3 code/visualize/entropy.py    "data/processed/$lang-full.csv" $lang Gemini gemini_min  --l-bins $L_BINS --output $OUT_FOLDER/entropy-$lang-gemini.png  --font "$FONT" --vmax $VMAX

    echo ""
done

echo "=== Charting: en-human (exception due to data volume) ==="
# python3 code/visualize/hue_unique.py "data/mlmc/en-hue-mlmc.csv"  en Human human_min --n-bins $N_BINS --output $OUT_FOLDER/unique-en-human.png  --font "$FONT" --ymax 95
python3 code/visualize/entropy.py    "data/mlmc/en-full-mlmc.csv" en Human human_min --l-bins $L_BINS --output $OUT_FOLDER/entropy-en-human.png --font "$FONT" --vmax 5

echo "Done."