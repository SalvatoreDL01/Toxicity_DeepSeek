import os.path

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

from Configurations import PROMPTS_DIR, TRAIN_DIR, TRAIN_MODEL_NAME, TRAIN_MODEL_PATH
from DAPT.Generation import free_model


def make_database(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split text into chunks (e.g., 512 tokens each for efficient processing)
    chunk_size = 512
    chunks = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Create a Dataset
    dataset = Dataset.from_dict({"text": chunks})
    return dataset


def map_tokens(tokenizer, dataset):
    # Assign a pad token to the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a new padding token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    # Data collator for CLM
    from transformers import DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return tokenized_dataset, data_collator


def train_model(model_path, database_path, dir):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add pad token

    # load model on CPU (safe)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")

    model.resize_token_embeddings(len(tokenizer))

    dataset = make_database(database_path)
    tokenized_dataset, data_collator = map_tokens(tokenizer, dataset)

    logging_dir = os.path.join(dir, "logging")
    output_dir = os.path.join(dir, "output")
    model_dir = os.path.join(dir, "model")
    for d in [logging_dir, output_dir, model_dir]:
        os.makedirs(d, exist_ok=True)

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        overwrite_output_dir=True,
        num_train_epochs=3,
        save_steps=10_000,
        save_total_limit=2,
        learning_rate=5e-5,
        fp16=False,
        logging_steps=500,
        logging_dir=logging_dir,
        output_dir=output_dir,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    return model, tokenizer


def main():
    database_path_toxic = os.path.join(PROMPTS_DIR, 'toxic.txt')
    database_path_not_toxic = os.path.join(PROMPTS_DIR, 'not_toxic.txt')
    output_dir_toxic = os.path.join(TRAIN_DIR, TRAIN_MODEL_NAME + "_toxic")
    output_dir_not_toxic = os.path.join(TRAIN_DIR, TRAIN_MODEL_NAME + "_not_toxic")

    model, tokenizer = train_model(TRAIN_MODEL_PATH, database_path_toxic, output_dir_toxic)
    free_model(generator=None, model=model, tokenizer=tokenizer)

    model, tokenizer = train_model(TRAIN_MODEL_PATH, database_path_not_toxic, output_dir_not_toxic)
    free_model(generator=None, model=model, tokenizer=tokenizer)


if __name__ == '__main__':
    main()
