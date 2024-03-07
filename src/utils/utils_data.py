import json
import os
from itertools import chain


def load_train_splits(val_id, path_folds, path_data, path_extra):
    folds = os.listdir(path_folds)
    folds = [fold for fold in folds if fold.endswith('json')]
    train_folds = [fold for fold in folds if fold != "fold_" + str(val_id) + ".json"]
    data = []
    for fold in train_folds:
        data += json.load(open(os.path.join(path_folds, fold)))
        
    data += json.load(open(os.path.join(path_data, path_extra)))
    return data

def load_validation_split(val_id, path_folds):
    fold = "fold_"+str(val_id)+".json"
    data = json.load(open(os.path.join(path_folds, fold)))
    return data

def get_full_data(path_data):
    data = json.load(open(os.path.join(path_data, "train.json")))
    return data

def get_label_mapping(data):
    all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
    label2id = {l: i for i,l in enumerate(all_labels)}
    id2label = {v:k for k,v in label2id.items()}
    return all_labels, label2id, id2label

def df_to_dict_list(df):
    """
    Convert Pandas DataFrame to a list of dictionaries.

    Parameters:
        df (DataFrame): The input Pandas DataFrame.

    Returns:
        list: A list of dictionaries where each dictionary represents a row in the DataFrame.
    """
    # Convert DataFrame to list of dictionaries
    dict_list = df.to_dict(orient='records')
    return dict_list
