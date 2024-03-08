

from itertools import chain
from datasets import Dataset

from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

import pandas as pd
from utils.utils_config import *
from utils.utils_data import *
from utils.utils_test import *

import os
import csv


CONFIG = "../configs/config.yaml"
model_path = "/home/ybouaouni/workspace/Training/PII-Data-Detection-Transformers/src/deberta3base_debug"
VAL_ID = 3

config = get_config(CONFIG)

data = get_full_data(config.PATH_DATA)
all_labels, label2id, id2label = get_label_mapping(data)
validation_data = load_validation_split(VAL_ID, config.PATH_FOLDS)
tokenizer = AutoTokenizer.from_pretrained(model_path)
ds_validation = prepare_dataset_test(validation_data, tokenizer, config.INFERENCE_MAX_LENGTH, config.STRIDE)



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

preds_final = get_class_prediction(preds, config.THRESHOLD)
ignored_labels   = ("O")
processed, pairs = get_doc_token_pred_triplets(preds_final, Dataset.from_dict(ds_dict), id2label, ignored_labels)



df = pd.DataFrame(processed)
df["row_id"] = list(range(len(df)))
df[["row_id", "document", "token", "label"]].to_csv("submission_good.csv", index=False)
p= os.path.join(config.PATH_FOLDS, "fold_"+str(VAL_ID)+".csv")
df_gt = pd.read_csv(p)
precision, recall, f1 = compute_metrics_eval(df, df_gt)
print(precision, recall, f1)