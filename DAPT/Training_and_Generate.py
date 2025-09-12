import os.path
import torch
from DAPT.Generation import free_model, generate_continuation
from DAPT.Model_Training import train_model

TRAIN_DIR = '../DAPT/models'

labels = ['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0']

PROMPT_DIR = '../DataBase/Prompts'

TRAIN_MODEL_PATH = 'openai-gpt'
TRAIN_MODEL_NAME = 'GPT-1'

gen_params = {
    "max_new_tokens": 20,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 0.9,
    "num_return_sequences": 1,
    "do_sample": True,
    "repetition_penalty": 1.0,
}

OUTPUT_DIR = '../DataBase/Generated'
PROMPT_DIR = '../DataBase/Prompts'

labels = ['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0']
files = {label: f"{PROMPT_DIR}/prompts_{label}.csv" for label in labels}

def main():
    toxic_labels = ['toxic', 'not_toxic']

    for toxic_label in toxic_labels:

        database_path = os.path.join(PROMPT_DIR, f'{toxic_label}.txt')
        output_model_dir = os.path.join(TRAIN_DIR, TRAIN_MODEL_NAME + f"_{toxic_label}")
        print("#    Training model: ", TRAIN_MODEL_NAME)
        model, tokenizer = train_model(TRAIN_MODEL_NAME, TRAIN_MODEL_PATH, database_path, output_model_dir)
        print("#    Model trained successfully!")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        if device == "cuda":
            torch.cuda.set_device(0)
            print(f"Device set to cuda:{torch.cuda.current_device()}")

        output_dir = os.path.join(OUTPUT_DIR, TRAIN_MODEL_NAME+"_"+toxic_label)
        os.makedirs(output_dir, exist_ok=True)
        generate_continuation(device, model, tokenizer, output_dir, files)

        free_model(generator=None, model=model, tokenizer=tokenizer)


if __name__ == '__main__':
    main()