from typing import Literal

import pandas as pd
import ast
import os
import json

#todo creare configuration
#todo testare
DATABASE_URI = "hf://datasets/allenai/real-toxicity-prompts/prompts.jsonl"
PROMTS_DIR = "../DataBase/Prompts"
# Definisci intervalli e labels
bins = [0.0, 0.25, 0.5, 0.75, 1.0]
labels = ['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0']
size = 6250


def safe_parse(x):
    """Try to parse dict-like strings safely."""
    if isinstance(x, dict):
        return x  # already a dict
    try:
        return ast.literal_eval(x)   # safest eval
    except Exception:
        try:
            return json.loads(x.replace("'", '"'))  # sometimes single vs double quotes
        except Exception:
            return {}  # fallback empty dict


def database_to_dataframe(df: pd.DataFrame, column: Literal["continuation", "prompt"]):
    # Suppose your dataframe is df with column "prompts"
    df["parsed"] = df[column].apply(safe_parse)

    # Flatten dicts into columns
    df_final = pd.json_normalize(df["parsed"])
    df_final = df_final.rename(columns={'text': 'Prompts'})

    return df_final


def saving_databse_DAPT(df: pd.DataFrame, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in df["Prompts"]:
            f.write(str(item) + "\n")


def saving_ranges(bins: list[int], labels: list[str], size: int, df: pd.DataFrame, output_dir: str):
    min = size
    df["toxicity_bin"] = pd.cut(df["toxicity"], bins=bins, labels=labels, include_lowest=True)

    # Output directory
    os.makedirs(output_dir, exist_ok=True)

    # Saving each range in a different csv
    for label in labels:
        subset = df[df["toxicity_bin"] == label]

        subset_size = len(subset)

        if size < subset_size:
            subset = subset.head(size)
        else:
            if subset_size < min:
                min = subset_size

        out_path = os.path.join(output_dir, f"prompts_{label}.csv")
        subset.to_csv(out_path, index=False)

        print(f"Saved: {out_path} ({len(subset)} rows)")

    return min


def resize_prompts(labels: list[str], min_size: int, PROMPTS_DIR: str):
    """
    Resizing of the CSV in PROMPTS_DIR.

    Args:
        labels (list): labels of the ranges
        min_size (int): number of rows from the smallest CSV
        PROMPTS_DIR (str): CSV directory
    """
    for label in labels:
        file_path = os.path.join(PROMPTS_DIR, f"prompts_{label}.csv")

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}, Skiped.")
            continue

        df = pd.read_csv(file_path)
        original_len = len(df)

        if original_len > min_size:
            df_resized = df.sample(n=min_size, random_state=42).reset_index(drop=True)
            df_resized.to_csv(file_path, index=False)
            print(f"{label}: resized from {original_len} to {min_size} rows")
        else:
            print(f"{label}: has {original_len} rows, no resizing needed (less than {min_size})")



def main():
    # Reading of the selected DataBase real-toxicity-prompts in this case
    df = pd.read_json(DATABASE_URI, lines=True)
    print("Loaded DataBase.")

    # Extraction of the promts and the continuation with their respective scores
    df_prompts = database_to_dataframe(df.copy(), 'prompt')
    df_continuation = database_to_dataframe(df.copy(), 'continuation')

    # Creation of the dataframe for the DAPT
    df_toxic = df_continuation[df_continuation['toxicity'] >= 0.5]
    df_not_toxic = df_continuation[df_continuation['toxicity'] < 0.5]

    # Resizing the dimension of the dataframes for the DAPT
    if len(df_toxic) > len(df_not_toxic):
        df_toxic = df_toxic.head(len(df_not_toxic))
    else:
        df_not_toxic = df_not_toxic.head(len(df_toxic))

    os.makedirs(PROMTS_DIR, exist_ok=True)
    saving_databse_DAPT(df_not_toxic, PROMTS_DIR+"/not_toxic.txt")
    print(f"Saved DAPT not toxic DataBase at: {PROMTS_DIR}/not_toxic.txt")
    saving_databse_DAPT(df_toxic, PROMTS_DIR+"/toxic.txt")
    print(f"Saved DAPT toxic DataBase at: {PROMTS_DIR}/toxic.txt")

    df_prompts.to_csv(PROMTS_DIR+"/Promts.csv", index=False)
    print(f"Saved Prompts at: {PROMTS_DIR}/Promts.csv")

    min = saving_ranges(bins, labels, size, df_prompts, PROMTS_DIR)
    if min < size:
        resize_prompts(bins, labels, min, PROMTS_DIR)

    print("Prompt generation ended.")


if __name__ == "__main__":
    main()
