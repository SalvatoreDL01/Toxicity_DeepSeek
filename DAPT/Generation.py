import gc
import os
import pandas as pd
import psutil
from datasets import Dataset
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from Configurations import mapped_trained_models, gen_params, PROMPTS_DIR, labels, OUTPUT_DIR


def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / (1024 ** 2)  # in MB
    if torch.cuda.is_available():
        gpu_mem_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print(
            f"[{note}] CPU: {cpu_mem:.2f} MB | GPU allocated: {gpu_mem_alloc:.2f} MB | reserved: {gpu_mem_reserved:.2f} MB")
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


def generate_continuation(device, model, tokenizer, output_dir, files):
    # Batch size (tune depending on VRAM)
    batch_size = 8 if device == "cuda" else 1

    for label, input_file in files.items():
        print(f"\nProcessing file: {input_file} (label: {label})")

        # Load prompts as Dataset
        df = pd.read_csv(input_file)
        dataset = Dataset.from_pandas(df)

        output_file = f"{output_dir}/responses_{label}.csv"

        # Initialize output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Prompts,Continuation\n")

        # Iterate in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Processing {label}"):
            batch = dataset[i:i + batch_size]["Prompts"]

            # Tokenize batch
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

            # Generate continuations
            gen_params['pad_token_id'] = tokenizer.eos_token_id
            outputs = model.generate(
                **inputs,
                **gen_params,
            )

            # Decode results
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Estrarre solo la continuation (senza prompt iniziale)
            results = []
            for prompt, full_text in zip(batch, decoded):
                continuation = full_text[len(prompt):].replace("\n", " ").strip()
                results.append((prompt, continuation))

            # Save batch results
            with open(output_file, "a", encoding="utf-8") as f:
                for prompt, cont in results:
                    f.write(f'"{prompt}","{cont}"\n')

        # Free GPU memory between files
        if device == "cuda":
            torch.cuda.empty_cache()

    print(" All files processed successfully!\n\n")


def main():
    files = {label: f"{PROMPTS_DIR}/prompts_{label}.csv" for label in labels}
    # Detect GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        torch.cuda.set_device(0)
        print(f"Device set to cuda:{torch.cuda.current_device()}")

    # Load fine-tuned model
    for model_name in mapped_trained_models:
        model_path = mapped_trained_models[model_name]
        print("# Loading model: " + model_name)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        set_seed(42)
        print("# Loaded successfully")

        output_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(output_dir, exist_ok=True)

        generate_continuation(device, model, tokenizer, output_dir, files)

        free_model(generator=None, model=model, tokenizer=tokenizer)


if __name__ == '__main__':
    main()
