import pandas as pd
import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from transformers import AutoTokenizer

from Text_manipulation.Cleaning import remove_overlap_from_str2, truncate_to_20_tokens, remove_boilerplate_continuation

# Thread-safe CSV writing
csv_lock = Lock()


DEEPSEEK_API_KEY = 'INSERT YOU API KEY'
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

labels = ['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0']

PROMPT_DIR = '../DataBase/Prompts'
OUTPUT_DIR = '../DataBase/Generated'

DATA = {
    "model": "deepseek-chat",
    "temperature": 1,
    "max_tokens": 50
}

n_required = 6250
max_workers = 5

# ✅ Tokenizer DeepSeek (più adatto del GPT-2)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")

def get_deepseek_response(prompt):
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    DATA["messages"] = [{"role": "user", "content": f"Continue this text: {prompt}"}]

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=DATA, timeout=30)
        if response.status_code == 200:
            continuation = response.json()["choices"][0]["message"]["content"]

            # Cleaning steps
            continuation = remove_overlap_from_str2(prompt, continuation)
            continuation = remove_boilerplate_continuation(continuation)
            continuation = truncate_to_20_tokens(prompt, continuation, tokenizer=tokenizer , max_new_tokens=20)

            return prompt, continuation
        else:
            print(f"Error ({prompt[:20]}...): HTTP {response.status_code}")
            return prompt, None
    except Exception as e:
        print(f"Failed ({prompt[:20]}...): {str(e)}")
        return prompt, None


def save_to_csv(prompt, response_text, csv_file):
    new_row = pd.DataFrame({
        "Prompts": [prompt],
        "Continuation": [response_text],
        "Timestamp": [pd.Timestamp.now()]
    })

    with csv_lock:
        header = not os.path.exists(csv_file)
        new_row.to_csv(
            csv_file,
            mode='a',
            header=header,
            index=False,
            lineterminator='\n'
        )


def process_prompt(prompt, csv_file):
    prompt_text, response_text = get_deepseek_response(prompt)
    if response_text:
        save_to_csv(prompt_text, response_text, csv_file)
        return True
    return False


def generate_responses_for_range(input_csv, output_csv, n_required=6250, max_workers=5):
    prompts = pd.read_csv(input_csv)["Prompts"].tolist()

    if os.path.exists(output_csv):
        used_prompts = set(pd.read_csv(output_csv)["Prompts"].tolist())
    else:
        used_prompts = set()

    prompts = [p for p in prompts if p not in used_prompts]
    success_count = len(used_prompts)

    print(f"\n=== Processing range file: {input_csv} ===")
    print(f"Already processed {success_count}/{n_required} in {output_csv}")

    while success_count < n_required and prompts:
        remaining = prompts[:n_required - success_count]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_prompt, prompt, output_csv): prompt for prompt in remaining}

            for future in as_completed(futures):
                prompt = futures[future]
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                        print(f"Success ({success_count}/{n_required}) - {prompt[:40]}...")
                except Exception as e:
                    print(f"Exception on {prompt[:20]}... : {e}")

        if success_count < n_required:
            print(f" Retry needed, collected {success_count}/{n_required}")
            time.sleep(5)

    print(f"Completed {success_count}/{n_required} for {input_csv}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for label in labels:
        input_file = os.path.join(PROMPT_DIR, f"prompts_{label}.csv")
        if not os.path.exists(input_file):
            print(f"#   File {input_file} not found. Skipping...")
            continue

        output_dir = os.path.join(OUTPUT_DIR, "DeepSeek_API")
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"responses_{label}.csv")

        print(f"#   Processing {input_file} → {output_file}")
        generate_responses_for_range(
            input_file,
            output_file,
            n_required=n_required,
            max_workers=max_workers
        )


if __name__ == "__main__":
    main()
