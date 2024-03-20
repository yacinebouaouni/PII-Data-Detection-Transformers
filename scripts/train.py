"""Module for training a token classification model."""

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "piidetect"))
)
from transformers import AutoTokenizer, Trainer
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)

from data import DatasetPII
from utils.utils_train import prepare_dataset
from utils.utils_config import get_config, set_random_seeds, get_trainer_args
import argparse

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
    initializing the model and tokenizer, creating the data collator,
    and initializing the Trainer object for training. It then starts the training process
    and saves the trained model
    """

    dataset = DatasetPII(cross_val=False, config=config)
    set_random_seeds(config.SEED)
    args = get_trainer_args(config)

    train_data = dataset.load_train_data()
    all_labels, label2id, id2label = (
        dataset.all_labels,
        dataset.label2id,
        dataset.id2label,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.TRAINING_MODEL_PATH)

    ds = prepare_dataset(
        train_data,
        tokenizer=tokenizer,
        all_labels=all_labels,
        label2id=label2id,
        id2label=id2label,
        max_length=config.TRAINING_MAX_LENGTH,
        seed=config.SEED,
    )

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
    model = AutoModelForTokenClassification.from_pretrained(
        config.TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("start training")
    trainer.train()

    print("saving model")
    if config.SAVE_MODELS:
        trainer.save_model(config.PATH_SAVE)
        tokenizer.save_pretrained(config.PATH_SAVE)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train the model with custom configurations"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of training epochs"
    )
    parser.add_argument("--batch", type=int, required=True, help="Batch size")
    parser.add_argument(
        "--accumulation", type=int, required=True, help="Gradient accumulation steps"
    )
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    config = get_config(CONFIG)
    config.TRAINING_MODEL_PATH = "microsoft/deberta-v3-large"
    config.SAVE_MODELS = True

    config.EPOCHS = args.epochs
    config.BATCH = args.batch
    config.ACCUMULATION = args.accumulation
    config.LR = args.lr

    config.WARMUP = 0.1
    N_GPU = 2
    dataset_name = "nicholas"  # ['tonyarobertson', 'mpware', 'nicholas', 'moth', 'pjmathematician']
    config.EXTRA_DATA = [dataset_name]
    config.PATH_SAVE = f"../models/deberta_large_{dataset_name}_{config.EPOCHS}_{config.BATCH*N_GPU}_{config.ACCUMULATION}_{config.LR}"
    print(f"Dataset = {config.EXTRA_DATA}, Model = {config.PATH_SAVE}")
    train(config)
