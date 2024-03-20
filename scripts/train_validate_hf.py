"""
cross_validation.py - Script for cross-validation for PII detection using transformers.

This script trains a token classification model for detecting personally identifiable information (PII)
in text using the transformers library. It performs cross-validation with the specified threshold value for evaluation.

Usage: python cross_validation.py --threshold <threshold_value>

Args:
    --threshold (float): Threshold value used for evaluation of the trained model.

"""


import os
import sys
import argparse
import gc
import statistics
from datetime import datetime

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "piidetect"))
)


import mlflow
from piidetect.data import DatasetPII
from piidetect.utils.utils_config import get_config, get_trainer_args, set_random_seeds
from piidetect.utils.utils_test import prepare_dataset_test, test
from piidetect.utils.utils_train import prepare_dataset, compute_metrics
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
)

from functools import partial


CONFIG = "../configs/config.yaml"
EXPERIMENTS_PATH = "experiments-deberta-base-datasets"


def train(config):
    """
    Train function to train the PII detection model.

    Args:
        config (Config): Configuration object containing model training parameters.

    Returns:
        None
    """
    set_random_seeds(config.SEED)
    args = get_trainer_args(config)

    current_datetime = datetime.now()
    run_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    # setup mlflow uri and run
    mlflow.set_tracking_uri(EXPERIMENTS_PATH)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(config.to_dict())

    f1_list = []
    for i in range(config.FOLDS):
        metrics = train_(args, i, config)
        # log CV metrics and end run
        mlflow.log_metric(f"f1_fold_{i}", metrics["f1"])
        mlflow.log_metric(f"precision_fold_{i}", metrics["precision"])
        mlflow.log_metric(f"recall_fold_{i}", metrics["recall"])
        f1_list.append(metrics["f1"])

    f1_mean = sum(f1_list) / len(f1_list)
    mlflow.log_metric("mean_f1", f1_mean)
    mlflow.log_metric("std_f1", statistics.pstdev(f1_list))

    mlflow.end_run()


def train_(args, val_id, config):
    """
    Train function to train the PII detection model for a single fold.

    Args:
        args (dict): Trainer arguments.
        val_id (int): Fold ID for validation.
        config (Config): Configuration object containing model training parameters.

    Returns:
        dict: Metrics including precision, recall, and f1 score.
    """

    dataset = DatasetPII(cross_val=True, config=config)
    train_data = dataset.load_train_splits(val_id)[:3000]
    validation_data = dataset.load_validation_split(val_id)[:500]

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

    ds_validation = prepare_dataset_test(
        validation_data,
        tokenizer=tokenizer,
        max_length=config.TRAINING_MAX_LENGTH,
        stride=config.STRIDE,
    )

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
    model = AutoModelForTokenClassification.from_pretrained(
        config.TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    model.dropout.p = config.DROPOUT

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=ds_validation,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, all_labels=all_labels),
    )

    trainer.train()

    # Delete the model and tokenizer objects
    del trainer
    del tokenizer

    # Call the garbage collector to free memory
    gc.collect()
    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    datasets = ["tonyarobertson", "mpware", "nicholas", "moth", "pjmathematician"]
    datasets = ["nicholas"]
    config = get_config(CONFIG)
    config.TRAINING_MODEL_PATH = "microsoft/deberta-v3-base"

    for dataset in datasets:
        for BATCH in [4, 6]:
            for DROPOUT in [0.15, 0.1]:
                config.EXTRA_DATA = [dataset]
                config.BATCH = BATCH
                config.DROPOUT = DROPOUT

                train(config)
