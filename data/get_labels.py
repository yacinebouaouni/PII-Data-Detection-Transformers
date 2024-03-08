import json
import os 
import pandas as pd

SPLITS = 4 
PATH = "data/kfold"

def save_labels(fold, save_path):
    labels = []
    for doc in fold:
        for token_idx, token_label in enumerate(doc['labels']):
            if token_label != "O":
                labels.append({"document": doc["document"], 
                            "token": token_idx, 
                            "label": token_label, 
                            "token_str": doc["tokens"][token_idx]})
    df = pd.DataFrame(labels)
    df["row_id"] = list(range(len(df)))
    df[["row_id", "document", "token", "label"]].to_csv(save_path, index=False)


for i in range(SPLITS):
    fold_str = "fold_"+str(i)
    fold = json.load(open(os.path.join("data/kfold", fold_str+".json")))
    save_path = os.path.join(PATH, fold_str+'.csv')
    save_labels(fold, save_path)