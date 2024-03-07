import json
import numpy as np
from functools import partial

from itertools import chain
from datasets import Dataset

import torch
from transformers import set_seed
from transformers import AutoTokenizer, Trainer
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
import evaluate

from utils.utils_config import *
from utils.utils_data import *
from utils.utils_test import *
from utils.utils_train import *

import pandas as pd
import os
import csv

import mlflow
import argparse
from datetime import datetime
import statistics 


CONFIG = "../configs/config.yaml"
print(os.getcwd())

def train(config):

    set_random_seeds(config.SEED)
    current_datetime = datetime.now()
    run_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    args = get_trainer_args(config)
    
    # setup mlflow uri and run
    mlflow.set_tracking_uri("Experiment")
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(config.config)
    
    # launch training/validation on each fold
    eval_f1_folds = []    
    for i in range(config.FOLDS):
        f1 = train_(i, args, config)
        eval_f1_folds.append(f1)
    mean_f1 = sum(eval_f1_folds)/len(eval_f1_folds)
    
    #log CV metrics and end run
    mlflow.log_metric("cv_f1", mean_f1)
    mlflow.log_metric("std_f1", statistics.pstdev(eval_f1_folds))
    mlflow.end_run()


def train_(val_id, args, config):
    fold_str = f"_fold_{val_id}"
    train_data = load_train_splits(val_id, config.PATH_FOLDS, config.PATH_DATA, config.PATH_NICHOLAS)
    validation_data = load_validation_split(val_id, config.PATH_FOLDS)
    data = get_full_data(config.PATH_DATA)
    all_labels, label2id, id2label = get_label_mapping(data)
    tokenizer = AutoTokenizer.from_pretrained(config.TRAINING_MODEL_PATH)

    ds_train = prepare_dataset(train_data, tokenizer, all_labels, label2id, id2label, config.TRAINING_MAX_LENGTH, config.SEED)
    ds_validation = prepare_dataset_test(validation_data, tokenizer, config.INFERENCE_MAX_LENGTH, config.STRIDE)
    
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
    model = AutoModelForTokenClassification.from_pretrained(
        config.TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
        )

    trainer = Trainer(
            model=model, 
            args=args, 
            train_dataset=ds_train, 
            #eval_dataset=ds_validation, 
            data_collator=collator, 
            tokenizer=tokenizer,
            compute_metrics=partial(compute_metrics, all_labels=all_labels),
        )
    

    print(f'start training for evaluation fold = {val_id}')
    training_output = trainer.train()
    ignored_labels   = ("O")

    precision, recall, f1 = test(trainer, ds_validation, config.STRIDE, config.THRESHOLD, config.PATH_FOLDS, ignored_labels, id2label, val_id)

    # Log metrics
    mlflow.log_metric("precision_"+fold_str, precision)
    mlflow.log_metric("recall_"+fold_str, recall)
    mlflow.log_metric("f1_"+fold_str, f1)


    if config.SAVE_MODELS:
        trainer.save_model(config.MODEL_SAVE)
        tokenizer.save_pretrained(config.MODEL_SAVE)
    
    return f1

    

    
if __name__ == "__main__":
    
    config = get_config(CONFIG)
    train(config)
    
    """
    for EPOCHS in [2]:
        for BATCH in [12]:
            for ACCUMULATION in [1]:
                print(f'Training for Epochs = {EPOCHS}, BATCH={BATCH}, ACCUMULATION={ACCUMULATION}')
                train()
    """