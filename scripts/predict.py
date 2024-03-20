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
from metric.metric import compute_metrics
import pandas as pd
from data import DatasetPII


def get_preds_model_(model_path, tokenizer, config):
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


CONFIG = "../configs/config.yaml"
VAL_ID = 0

model_path = "/home/ybouaouni/workspace/Training/PII-Data-Detection-Transformers/models/top_models_957/deberta3base_nicholas_ep2_bs6_acc1"

config = get_config(CONFIG)
config.STRIDE = 384
config.THRESHOLD = 0.95


dataset = DatasetPII(cross_val=True, config=config)
all_labels, label2id, id2label = dataset.all_labels, dataset.label2id, dataset.id2label
validation_data = dataset.load_validation_split(VAL_ID)
tokenizer = AutoTokenizer.from_pretrained(model_path)
ds_validation = prepare_dataset_test(
    validation_data, tokenizer, config.INFERENCE_MAX_LENGTH, config.STRIDE
)


print("predictions of model 1")
preds, ds_dict = get_preds_model_(
    model_path=model_path, tokenizer=tokenizer, config=config
)


preds_final = get_class_prediction(preds, config.THRESHOLD)

ignored_labels = "O"
processed, pairs = get_doc_token_pred_triplets(
    preds_final, Dataset.from_dict(ds_dict), id2label, ignored_labels
)


df = pd.DataFrame(processed)
df["row_id"] = list(range(len(df)))
df.to_csv(f"predictions_fold{VAL_ID}.csv")
df_gt = pd.read_csv(os.path.join(config.PATH_FOLDS, "fold_" + str(VAL_ID) + ".csv"))
precision, recall, f1 = Evaluator.compute_metrics_eval(df, df_gt)

print("---metric2---")

results = compute_metrics(df, df_gt)
print(results)
