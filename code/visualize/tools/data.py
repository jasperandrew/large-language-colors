import pandas as pd


def load_and_validate(csv_path: str, term_col: str):
    df = pd.read_csv(csv_path)
    missing = {"r", "g", "b", term_col} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df