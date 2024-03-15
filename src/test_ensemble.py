from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)

from utils.utils_config import *
from utils.utils_data import *
from utils.utils_test import *

import pandas as pd
import os


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
model_path = "/home/ybouaouni/workspace/Training/top_models_957/deberta3base_nicholas_ep2_bs6_acc1"
model_path2 = (
    "/home/ybouaouni/workspace/Training/top_models_957/deberta3base_mpware_ep2_bs6_acc1"
)
model_path3 = "/home/ybouaouni/workspace/Training/deberta_nicholas_ep2_bs8"

VAL_ID = 0

config = get_config(CONFIG)

data = get_full_data(config.PATH_DATA)
all_labels, label2id, id2label = get_label_mapping(data)
validation_data = load_validation_split(VAL_ID, config.PATH_FOLDS)
tokenizer = AutoTokenizer.from_pretrained(model_path)
ds_validation = prepare_dataset_test(
    validation_data, tokenizer, config.INFERENCE_MAX_LENGTH, config.STRIDE
)


print("predictions of model 1")
preds1, ds_dict = get_preds_model_(
    model_path=model_path, tokenizer=tokenizer, config=config
)
print("predictions of model 2")
preds2, ds_dict = get_preds_model_(
    model_path=model_path2, tokenizer=tokenizer, config=config
)
print("predictions of model 3")
preds3, ds_dict = get_preds_model_(
    model_path=model_path3, tokenizer=tokenizer, config=config
)

preds = []
for i, (p1, p2, p3) in enumerate(zip(preds1, preds2, preds3)):
    preds.append((p1 + p2 + p3) / 3)


preds_final = get_class_prediction(preds, config.THRESHOLD)
ignored_labels = "O"
processed, pairs = get_doc_token_pred_triplets(
    preds_final, Dataset.from_dict(ds_dict), id2label, ignored_labels
)


df = pd.DataFrame(processed)
df["row_id"] = list(range(len(df)))
# df[["row_id", "document", "token", "label"]].to_csv("submission_ensemble.csv", index=False)
p = os.path.join(config.PATH_FOLDS, "fold_" + str(VAL_ID) + ".csv")
df_gt = pd.read_csv(p)
precision, recall, f1 = compute_metrics_eval(df, df_gt)
print(precision, recall, f1)
