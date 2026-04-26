# Large Language Colors: Variation and Misalignment in LLM Color Naming
Jasper Andrew (andrewj2@montclair.edu)

## Replication Instructions

### Environment

1. Install `python==3.14.2` (or use a conda environment, or whatever you like)
2. Run `python -m ensurepip --upgrade`
3. Run `pip install -r requirements.txt`
4. Create `code/collect/tools/api_keys.py` with your Google and OpenAI API keys:
    ```
    GOOGLE_API_KEY = "<key>"
    OPENAI_API_KEY = "<key>"
    ```
5. All main scripts have help text, run `python <file> -h` for usage info

### MLMC Data (as of 05/2026)
1. Navigate to [the MLMC repository's cleaned data folder](https://github.com/uwdata/color-naming-in-different-languages/tree/master/model/cleaned_color_data_by_lang)
2. Download `cleaned_color_names-XX.csv` for whichever language(s) you are analyzing
3. Run `extract_mlmc.py --rgbset <RGBSET> cleaned_color_names-XX.csv <OUT_CSV>`
    - `RGBSET`: "full" for full-cube data or "line" for hue-line data
    - `OUT_CSV`: the filepath to output to (will overwrite)

### General Method
1. Collect (`code/collect/query.py`)
2. Process (`code/process/normalize.py`)
3. Analyze/Visualize (`code/analyze/...`, `code/visualize/...`)

## Acknowledgements

Huge thanks to the authors and maintainers of the **Many Languages, Many Colors** project! ([website](https://idl.uw.edu/color-naming-in-different-languages)) ([github](https://github.com/uwdata/color-naming-in-different-languages/))