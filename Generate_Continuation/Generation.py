import pandas as pd
from transformers import pipeline, set_seed
from tqdm import tqdm
import os
import torch
import gc
import psutil
import re

from Text_manipulation.Cleaning import clean_continuation

from Configurations import BAD_WORDS_PATH, PROMPTS_DIR, labels, mapped_models, OUTPUT_DIR, gen_params, \
    bad_word_mapped_model


def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / (1024 ** 2)  # in MB
    if torch.cuda.is_available():
        gpu_mem_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print(f"[{note}] CPU: {cpu_mem:.2f} MB | GPU allocated: {gpu_mem_alloc:.2f} MB | reserved: {gpu_mem_reserved:.2f} MB")
    else:
        print(f"[{note}] CPU: {cpu_mem:.2f} MB | GPU not available")

def free_model(generator=None, model=None, tokenizer=None):
    if generator is not None:
        del generator
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def process_model(model_name, model_path, files, OUTPUT_DIR, gen_params, flag_BW):

    print_memory_usage("Before loading")
    generator = pipeline("text-generation", model=model_path, device=0 if torch.cuda.is_available() else -1)
    print_memory_usage("After loading")

    set_seed(42)

    # Add model-specific params
    gen_params["pad_token_id"] = generator.tokenizer.eos_token_id

    if flag_BW:
        print("Removing tokens in: " + BAD_WORDS_PATH)
        with open(BAD_WORDS_PATH) as f:
            bad_words = f.read().splitlines()
        bad_words_ids = [
            generator.tokenizer(bw, add_special_tokens=False).input_ids
            for bw in bad_words
        ]
        gen_params["bad_words_ids"] = bad_words_ids
        output_dir = os.path.join(OUTPUT_DIR, model_name + '_BW')
    else:
        output_dir = os.path.join(OUTPUT_DIR, model_name)

    os.makedirs(output_dir, exist_ok=True)

    try:
        for label, input_file in files.items():
            print(f"#   Processing file {input_file} for model {model_name}")
            df = pd.read_csv(input_file)
            prompts = df["Prompts"].tolist()

            output_file = os.path.join(output_dir, f"responses_{label}.csv")

            # Initialize output file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("Prompts,Continuation\n")

            batch_size = 16  # tune depending on VRAM
            all_results = []

            for i in tqdm(range(0, len(prompts), batch_size), desc=f"{model_name} - {label}"):
                batch = prompts[i:i + batch_size]
                outputs = generator(batch, **gen_params)

                # outputs is nested list if batch_size > 1
                for prompt, outs in zip(batch, outputs):
                    generated_texts = [
                        clean_continuation(out["generated_text"][len(prompt):])
                        for out in (outs if isinstance(outs, list) else [outs])
                    ]
                    for text in generated_texts:
                        all_results.append((prompt, text))

            # Save all results at once
            results_df = pd.DataFrame(all_results, columns=["Prompts", "Continuation"])
            results_df.to_csv(output_file, mode="a", index=False, header=False, encoding="utf-8")

    except KeyboardInterrupt:
        print("\n⏸ Interrupted! Saving partial results...")
        if "all_results" in locals() and all_results:
            results_df = pd.DataFrame(all_results, columns=["Prompts", "Continuation"])
            results_df.to_csv(output_file, mode="a", index=False, header=False, encoding="utf-8")
        raise

    finally:
        free_model(generator)
        print_memory_usage("After freeing")


def main():
    print(' Generating continuations with the following models:')

    files = {label: f"{PROMPTS_DIR}/prompts_{label}.csv" for label in labels}

    # tqdm over models
    for model_name, model_path in tqdm(mapped_models.items(), desc="Deploying models", unit="model"):
        process_model(model_name, model_path, files, OUTPUT_DIR, gen_params, False)

    for model_name, model_path in tqdm(bad_word_mapped_model.items(), desc="Deploying models without bad words", unit="model"):
        process_model(model_name, model_path, files, OUTPUT_DIR, gen_params, True)

    print(' All models successfully deployed!')

if __name__ == "__main__":
    main()
