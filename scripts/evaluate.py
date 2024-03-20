import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "piidetect"))
)
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)

from utils.utils_config import *
from utils.utils_test import *
from metric import Evaluator
import pandas as pd
import os
from data import DatasetPII
import pickle
import json


def save_pickle_preds(variable, file_path):
    """
    Save a variable to a pickle file.

    Parameters:
        variable: Any - The variable to be saved.
        file_path: str - The file path (including the file name) where the variable will be saved.

    Returns:
        None
    """
    try:
        with open(file_path, "wb") as f:
            pickle.dump(variable, f)
        print(f"Variable saved successfully to {file_path}")
    except Exception as e:
        print(f"Error occurred while saving the variable to {file_path}: {e}")


def get_preds_model_(model_path, tokenizer, config, ds_validation):
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
    args = TrainingArguments(
        ".",
        per_device_eval_batch_size=1,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    preds, ds_dict = predict_data(ds_validation, trainer, config.STRIDE)
    return preds, ds_dict


def test_(config, val_id):
    dataset = DatasetPII(cross_val=True, config=config)
    all_labels, label2id, id2label = (
        dataset.all_labels,
        dataset.label2id,
        dataset.id2label,
    )
    validation_data = json.load(open("../data/train.json"))[:5]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ds_validation = prepare_dataset_test(
        validation_data, tokenizer, config.INFERENCE_MAX_LENGTH, config.STRIDE
    )

    preds, ds_dict = get_preds_model_(
        model_path=model_path,
        tokenizer=tokenizer,
        config=config,
        ds_validation=ds_validation,
    )

    path_save = f"preds_fold_{val_id}"
    save_pickle_preds(preds, path_save)


model_path = "/home/ybouaouni/workspace/Training/PII-Data-Detection-Transformers/models/deberta_base_nicholas_2_8_1"


if __name__ == "__main__":
    CONFIG = "../configs/config.yaml"

    config = get_config(CONFIG)

    for val_id in range(config.FOLDS):
        test_(config, val_id)
