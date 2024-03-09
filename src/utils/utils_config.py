from config.config import Config
from transformers import TrainingArguments
import torch
import random 
import numpy as np


def get_config(config_path):
    config = Config(config_path)
    return config


def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The seed value to use for setting random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def get_trainer_args(config):
    args = TrainingArguments(
            output_dir=config.OUTPUT_DIR, 
            fp16=True,
            warmup_steps=config.WARMUP,
            learning_rate=2e-5,
            num_train_epochs=config.EPOCHS,
            per_device_train_batch_size=config.BATCH,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=config.ACCUMULATION,
            report_to="none",
            evaluation_strategy="no",
            save_strategy="no",
            save_total_limit=1,
            overwrite_output_dir=True,
            lr_scheduler_type='cosine',
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=100,
            weight_decay=0.01,
            seed = config.SEED,
        )
    return args