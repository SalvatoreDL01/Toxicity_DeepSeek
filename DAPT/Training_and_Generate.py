import os.path
import torch

from Configurations import PROMPTS_DIR, TRAIN_DIR, TRAIN_MODEL_NAME, labels, TRAIN_MODEL_PATH, OUTPUT_DIR
from DAPT.Generation import free_model, generate_continuation
from DAPT.Model_Training import train_model


def main():
    toxic_labels = ['toxic', 'not_toxic']
    files = {label: f"{PROMPTS_DIR}/prompts_{label}.csv" for label in labels}

    for toxic_label in toxic_labels:

        database_path = os.path.join(PROMPTS_DIR, f'{toxic_label}.txt')
        output_model_dir = os.path.join(TRAIN_DIR, TRAIN_MODEL_NAME + f"_{toxic_label}")
        print("#    Training model: ", TRAIN_MODEL_NAME)
        model, tokenizer = train_model(TRAIN_MODEL_PATH, database_path, output_model_dir)
        print("#    Model trained successfully!")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        if device == "cuda":
            torch.cuda.set_device(0)
            print(f"Device set to cuda:{torch.cuda.current_device()}")

        output_dir = os.path.join(OUTPUT_DIR, TRAIN_MODEL_NAME + "_" + toxic_label)
        os.makedirs(output_dir, exist_ok=True)
        generate_continuation(device, model, tokenizer, output_dir, files)

        free_model(generator=None, model=model, tokenizer=tokenizer)


if __name__ == '__main__':
    main()
