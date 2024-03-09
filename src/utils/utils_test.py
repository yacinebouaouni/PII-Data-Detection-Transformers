import os
import numpy as np
import pandas as pd
from datasets import Dataset
from metric.metric import compute_metrics_eval


def tokenize_test(example, tokenizer, max_length, stride):
    text = []
    token_map = []
    
    idx = 0
    for t, ws in zip(example["tokens"], example["trailing_whitespace"]):
        text.append(t)
        token_map.extend([idx]*len(t))
        if ws:
            text.append(" ")
            token_map.append(-1)
            
        idx += 1
        
    tokenized = tokenizer("".join(text), return_offsets_mapping=True, 
                          truncation=True, 
                          max_length=max_length, 
                          stride=stride, 
                          return_overflowing_tokens=True)
        
    return {
        **tokenized,
        "token_map": token_map,
    }

def prepare_dataset_test(data, tokenizer, max_length, stride):
    ds = Dataset.from_dict({
    "full_text": [x["full_text"] for x in data],
    "document": [x["document"] for x in data],
    "tokens": [x["tokens"] for x in data],
    "trailing_whitespace": [x["trailing_whitespace"] for x in data],
    "provided_labels": [x["labels"] for x in data],
    })

    ds = ds.map(tokenize_test, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "stride":stride}, num_proc=4)
    return ds


def backwards_map_preds(sub_predictions, max_len, i, stride):
    if max_len != 1: # nothing to map backwards if sequence is too short to be split in the first place
        if i == 0:
            # First sequence needs no SEP token (used to end a sequence)
            sub_predictions = sub_predictions[:,:-1,:]
        elif i == max_len-1:
            # End sequence needs to CLS token + Stride tokens 
            sub_predictions = sub_predictions[:,1+stride:,:] # CLS tokens + Stride tokens
        else:
            # Middle sequence needs to CLS token + Stride tokens + SEP token
            sub_predictions = sub_predictions[:,1+stride:-1,:]
    return sub_predictions

def backwards_map_(row_attribute, max_len, i, stride):
    # Same logics as for backwards_map_preds - except lists instead of 3darray
    if max_len != 1:
        if i == 0:
            row_attribute = row_attribute[:-1]
        elif i == max_len-1:
            row_attribute = row_attribute[1+stride:]
        else:
            row_attribute = row_attribute[1+stride:-1]
    return row_attribute


def predict_data(ds, trainer, stride):
    """
    Process the given dataset by making predictions for each split and re-assembling the results.
    
    Args:
        - ds: The hugging face dataset to be processed.
        - trainer: The model or trainer used for predictions.
        
    Returns:
        - preds (list): A list of predictions for each token and for each class
        - ds_dict (dict): A dictionary containing processed dataset information including document, tokens, token map, and offset mapping.
    """
    
    preds = []
    ds_dict = {
        "document": [],
        "token_map": [],
        "offset_mapping": [],
        "tokens": []
    }

    for row in ds:
        row_preds = []
        row_offset = []

        for i, y in enumerate(row["offset_mapping"]):
            x = Dataset.from_dict({
                "token_type_ids": [row["token_type_ids"][i]],
                "input_ids": [row["input_ids"][i]],
                "attention_mask": [row["attention_mask"][i]],
                "offset_mapping": [row["offset_mapping"][i]]
            })
            pred = trainer.predict(x).predictions
            row_preds.append(backwards_map_preds(pred, len(row["offset_mapping"]), i, stride))
            row_offset += backwards_map_(y, len(row["offset_mapping"]), i, stride)

        ds_dict["document"].append(row["document"])
        ds_dict["tokens"].append(row["tokens"])
        ds_dict["token_map"].append(row["token_map"])
        ds_dict["offset_mapping"].append(row_offset)

        p_concat = np.concatenate(row_preds, axis=1)
        preds.append(p_concat)

    return preds, ds_dict


def get_class_prediction(preds, threshold):
    """
     Generate class predictions from model predictions based on a given threshold.
     
     args
         - preds: predictions of the model of size (num_samples, num_tokens, num_classes)
         - threshold: threshold for keeping the 'O' prediction
    
     Returns
         - preds_final: predictions of size (num_samples, num_tokens)
    
    """
    preds_final = []
    for predictions in preds:
        predictions_softmax = np.exp(predictions) / np.sum(np.exp(predictions), axis = 2).reshape(predictions.shape[0],predictions.shape[1],1)
        predictions = predictions.argmax(-1)
        predictions_without_O = predictions_softmax[:,:,:12].argmax(-1)
        O_predictions = predictions_softmax[:,:,12]

        preds_final.append(np.where(O_predictions < threshold, predictions_without_O , predictions))
    return preds_final


def get_doc_token_pred_triplets(preds_final, ds, id2label, ignored_labels):
    """Extracts document-token-label triplets from predictions and dataset.

    Args:
        preds_final (list): List of predicted labels for tokens. (num_samples, num_tokens)
        ds (Dataset): Dataset (Hugging Face) containing token mapping, offset mapping, tokens, and documents.
        id2label (Dict): mapping between ID and Labels used to train the model.
        ignored_labels: list of ignored labels 
    Returns:
        tuple: A tuple containing:
            - processed (list): List of dictionaries, each representing a document-token-label triplet.
              Each dictionary contains keys 'document', 'token', 'label', and 'token_str' corresponding to document ID,
              token ID, predicted label, and token string respectively.
            - pairs (set): Set containing unique pairs of document ID and token ID extracted.
    """

    pairs = set()  # membership operation using set is faster O(1) than that of list O(n)
    processed = []
    for p, token_map, offsets, tokens, doc in zip(preds_final, ds["token_map"], ds["offset_mapping"], ds["tokens"], ds["document"]):
        for token_pred, (start_idx, end_idx) in zip(p[0], offsets):
            label_pred = id2label[token_pred]

            if start_idx + end_idx == 0: 
                continue

            if token_map[start_idx] == -1:
                start_idx += 1

            # ignore "\n\n"
            while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                start_idx += 1

            if start_idx >= len(token_map): 
                break

            token_id = token_map[start_idx]

            # ignore "O" predictions and whitespace preds
            # we alsog ignore "B-EMAIL", "B-PHONE-NUM" and "I-PHONE-NUM" which will be processed later
            if label_pred in ignored_labels or token_id == -1:
                continue

            pair = (doc, token_id)

            if pair not in pairs:
                processed.append({"document": doc, "token": token_id, "label": label_pred, "token_str": tokens[token_id]})
                pairs.add(pair)
                
    return processed, pairs


def test(trainer, ds_validation, stride, threshold, path_folds, ignored_labels, id2label, val_id):
        
    preds, ds_dict = predict_data(ds_validation, trainer, stride)
    preds_final = get_class_prediction(preds, threshold)

    processed, _ = get_doc_token_pred_triplets(preds_final, Dataset.from_dict(ds_dict), id2label, ignored_labels)

    df = pd.DataFrame(processed)    
    df_gt = pd.read_csv(os.path.join(path_folds, "fold_"+str(val_id)+".csv"))

    precision, recall, f1 = compute_metrics_eval(df, df_gt)
        
    return precision, recall, f1


