from typing import Optional, Tuple
from googleapiclient import discovery
import os
import pandas as pd
from tqdm import tqdm
import time
from googleapiclient.errors import HttpError

from Configurations import perspective_attributes, PERSPECTIVE_OUTPUT_DIR, PERSPECTIVE_API_KEY, model_dirs


def analyze_text(client, text):
    """Analyze text with Perspective API."""
    analyze_request = {
        "comment": {"text": text},
        "requestedAttributes": {attr.upper(): {} for attr in perspective_attributes},
    }

    for attempt in range(5):  # retry loop
        try:
            response = client.comments().analyze(body=analyze_request).execute()
            summary_scores = {
                attr: response["attributeScores"][attr.upper()]["summaryScore"]["value"]
                for attr in perspective_attributes
            }
            return summary_scores
        except HttpError as e:
            if e.resp.status == 429:
                wait_time = 2 ** attempt
                print(f"# !!! Rate limit hit. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            elif e.resp.status == 400:
                # Unsupported language
                return {attr: None for attr in perspective_attributes}
            else:
                raise
    return {attr: None for attr in perspective_attributes}

def process_model_dir(model_name, model_dir, client):
    print(f"\n Processing model: {model_name} in {model_dir}")

    for file_name in os.listdir(model_dir):
        if not file_name.endswith(".csv"):
            continue

        input_path = os.path.join(model_dir, file_name)

        output_dir = PERSPECTIVE_OUTPUT_DIR + model_name
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, file_name)

        print(f" Processing file: {file_name}")
        df = pd.read_csv(input_path, quotechar='"', escapechar="\\", on_bad_lines="skip")

        # Ensure attributes columns exist
        for attr in perspective_attributes:
            if attr not in df.columns:
                df[attr] = None

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{model_name}:{file_name}"):
            text = str(row["Continuation"])
            scores = analyze_text(client, text)
            for col, score in scores.items():
                df.at[idx, col] = score
            time.sleep(1.05)  # respect quota

        df.to_csv(output_path, index=False)
        print(f" Saved results to {output_path}")

def main():
    os.makedirs(PERSPECTIVE_OUTPUT_DIR, exist_ok=True)

    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=PERSPECTIVE_API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    )

    for model_name, model_dir in model_dirs.items():
        process_model_dir(model_name, model_dir, client)

    print("\n All models processed successfully!")

if __name__ == "__main__":
    main()
