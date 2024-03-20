import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "piidetect"))
)
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    get_cosine_schedule_with_warmup,
)

from transformers import get_scheduler
from piidetect.utils.utils_train import *

from piidetect.data import DatasetPII
from utils.utils_config import get_config, set_random_seeds
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch

CONFIG = "../configs/config.yaml"


def train(config):
    """
    Train the model using the provided configuration.

    Args:
        config (Config): Configuration object containing various parameters for training.

    Returns:
        None

    This function initializes the necessary components for training the model,
    including loading the dataset, setting random seeds, preparing the dataset,
    initializing the model and tokenizer, and then starts the training process.
    """

    accelerator = Accelerator()

    device = accelerator.device

    dataset = DatasetPII(cross_val=False, config=config)

    set_random_seeds(config.SEED)

    train_data = dataset.load_train_data()[:300]
    all_labels, label2id, id2label = (
        dataset.all_labels,
        dataset.label2id,
        dataset.id2label,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.TRAINING_MODEL_PATH)
    train_data = prepare_dataset(
        train_data,
        tokenizer=tokenizer,
        all_labels=all_labels,
        label2id=label2id,
        id2label=id2label,
        max_length=config.TRAINING_MAX_LENGTH,
        seed=config.SEED,
    ).with_format("torch")

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
    dataloader = DataLoader(train_data)

    model = AutoModelForTokenClassification.from_pretrained(
        config.TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    # Define the number of training steps based on the dataset size and the batch size
    num_epochs = config.EPOCHS
    num_training_steps = num_epochs * len(dataloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    from tqdm.auto import tqdm

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            print(loss)
            accelerator.backward(loss)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.MAX_GRAD_NORM
            )
            print(f"grad_norm = {grad_norm}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1),


if __name__ == "__main__":
    config = get_config(CONFIG)
    config.TRAINING_MODEL_PATH = "microsoft/deberta-v3-large"
    config.SAVE_MODELS = True
    config.EPOCHS = 1
    config.BATCH = 1
    config.LR = 5e-5
    config.WARMUP = 0.1
    config.MAX_GRAD_NORM = 100
    N_GPU = 1
    dataset_name = "nicholas"  # ['tonyarobertson', 'mpware', 'nicholas', 'moth', 'pjmathematician']
    config.EXTRA_DATA = [dataset_name]
    config.PATH_SAVE = f"../models/deberta_large_{dataset_name}_{config.EPOCHS}_{config.BATCH*N_GPU}_{config.ACCUMULATION}_{config.LR}"
    print(f"Dataset = {config.EXTRA_DATA}, Model = {config.PATH_SAVE}")
    train(config)
