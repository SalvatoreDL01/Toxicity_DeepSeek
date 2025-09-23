from typing import Literal

import pandas as pd
import ast
import os
import json

from Configurations import DATABASE_URI, PROMPTS_DIR, bins, labels, size


def safe_parse(x):
    """Try to parse dict-like strings safely."""
    if isinstance(x, dict):
        return x  # already a dict
    try:
        return ast.literal_eval(x)  # safest eval
    except Exception:
        try:
            return json.loads(x.replace("'", '"'))  # sometimes single vs double quotes
        except Exception:
            return {}  # fallback empty dict


def database_to_dataframe(df: pd.DataFrame, column: Literal["continuation", "prompt"]) -> pd.DataFrame:
    # Preserva l'indice originale
    original_index = df.index

    # Suppose your dataframe is df with column "prompts"
    df = df.copy()  # Crea una copia per evitare modifiche all'originale
    df["parsed"] = df[column].apply(safe_parse)

    # Flatten dicts into columns
    df_final = pd.json_normalize(df["parsed"])
    df_final = df_final.rename(columns={'text': 'Prompts'})

    # Ripristina l'indice originale
    df_final.index = original_index

    return df_final


def saving_databse_DAPT(df: pd.DataFrame, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in df["Continuation"]:
            f.write(str(item) + "\n")


def saving_ranges(bins: list[int], labels: list[str], size: int, df: pd.DataFrame, output_dir: str) -> int:
    min_size = size
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
            if subset_size < min_size:
                min_size = subset_size

        out_path = os.path.join(output_dir, f"prompts_{label}.csv")
        subset.to_csv(out_path, index=False)

        print(f"Saved: {out_path} ({len(subset)} rows)")

    return min_size


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


def add_prompt_to_continuation(df_continuation: pd.DataFrame, df_prompts: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge una colonna con il relativo prompt per ogni continuazione.

    Args:
        df_continuation: DataFrame con le continuazioni
        df_prompts: DataFrame con i prompts

    Returns:
        DataFrame con una colonna aggiuntiva 'prompt_text' contenente il prompt relativo
    """
    # Verifica che gli indici siano allineati
    if not df_continuation.index.equals(df_prompts.index):
        print("Warning: Gli indici non sono allineati. Verranno riallineati automaticamente.")
        # Riallinea gli indici preservando l'ordine originale delle continuazioni
        df_prompts_aligned = df_prompts.reindex(df_continuation.index)
    else:
        df_prompts_aligned = df_prompts

    # Crea una copia per non modificare l'originale
    df_continuation_with_prompt = df_continuation.copy()

    # Aggiungi la colonna con il prompt corrispondente
    df_continuation_with_prompt['Prompts'] = df_prompts_aligned['Prompts']

    # Verifica che non ci siano valori nulli
    null_count = df_continuation_with_prompt['Prompts'].isnull().sum()
    if null_count > 0:
        print(f"Warning: Trovati {null_count} prompt mancanti.")

    return df_continuation_with_prompt


def main():
    # Reading of the selected DataBase real-toxicity-prompts in this case
    df = pd.read_json(DATABASE_URI, lines=True)
    print("Loaded DataBase.")
    print(f"Numero totale di righe: {len(df)}")

    # Extraction of the promts and the continuation with their respective scores
    # Usa lo stesso DataFrame originale per entrambe le elaborazioni
    df_prompts = database_to_dataframe(df, 'prompt')
    df_continuation = database_to_dataframe(df, 'continuation')

    df_continuation = df_continuation.rename(columns={"Prompts": "Continuation"})

    print(f"Dimensioni df_prompts: {df_prompts.shape}")
    print(f"Dimensioni df_continuation: {df_continuation.shape}")
    print(f"Indici allineati: {df_prompts.index.equals(df_continuation.index)}")

    # Aggiungi il prompt relativo a ogni continuazione
    df_continuation_with_prompt = add_prompt_to_continuation(df_continuation, df_prompts)

    # Creation of the dataframe for the DAPT
    df_toxic = df_continuation[df_continuation['toxicity'] >= 0.5]
    df_not_toxic = df_continuation[df_continuation['toxicity'] < 0.5]

    print(f"Continuazioni tossiche: {len(df_toxic)}")
    print(f"Continuazioni non tossiche: {len(df_not_toxic)}")

    # Resizing the dimension of the dataframes for the DAPT
    if len(df_toxic) > len(df_not_toxic):
        df_toxic = df_toxic.head(len(df_not_toxic))
    else:
        df_not_toxic = df_not_toxic.head(len(df_toxic))

    os.makedirs(PROMPTS_DIR, exist_ok=True)

    # Saving DAPT DBs
    saving_databse_DAPT(df_not_toxic, PROMPTS_DIR + "/not_toxic.txt")
    print(f"Saved DAPT not toxic DataBase at: {PROMPTS_DIR}/not_toxic.txt")
    saving_databse_DAPT(df_toxic, PROMPTS_DIR + "/toxic.txt")
    print(f"Saved DAPT toxic DataBase at: {PROMPTS_DIR}/toxic.txt")

    # Saving original prompts with attributes
    df_prompts.to_csv(PROMPTS_DIR + "/Prompts.csv", index=False)
    print(f"Saved Prompts at: {PROMPTS_DIR}/Prompts.csv")

    # Saving original continuation with prompts and continuation attributes
    df_continuation_with_prompt.to_csv(PROMPTS_DIR + "/Continuation.csv", index=False)
    print(f"Saved Continuation with prompts at: {PROMPTS_DIR}/Continuation.csv")

    if size == 0:
        used_size = len(df_prompts)
    else:
        used_size = size

    min_size = saving_ranges(bins, labels, used_size, df_prompts, PROMPTS_DIR)
    if min_size < used_size:
        resize_prompts(labels, min_size, PROMPTS_DIR)

    print("Prompt generation ended.")


if __name__ == "__main__":
    main()