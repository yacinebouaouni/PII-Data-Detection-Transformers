from datasets import Dataset
import numpy as np


def prepare_dataset(data, tokenizer, all_labels, label2id, id2label, seed, max_length):
    ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [str(x["document"]) for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
        "provided_labels": [x["labels"] for x in data],
    }).shuffle(seed=seed)

    ds = ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": max_length}, num_proc=8)
    return ds


def tokenize(example, tokenizer, label2id, max_length):
    """
    Function to tokenize the text and map the old labels/tokens to the
    new tokens.
    """

    # Rebuild text from tokens with respective labels
    text = []
    labels = []

    for t, l, ws in zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"]):
        text.append(t)
        labels.extend([l] * len(t))

        if ws:
            text.append(" ")
            labels.append("O")
    
    text = "".join(text)
    # Tokenize the text with offset mapping to keep track of tokens positions
    tokenized = tokenizer(text, return_offsets_mapping=True, max_length=max_length, truncation=True)

    # Map labels from old tokens to new tokenization
    labels = np.array(labels)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:
        # CLS token
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue

        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1

        token_labels.append(label2id[labels[start_idx]])


    return {**tokenized, "labels": token_labels, "length": len(tokenized.input_ids)}

def compute_metrics(p, all_labels):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    recall = recall_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    f1_score = (1 + 5*5) * recall * precision / (5*5*precision + recall)
    
    results = {
        'recall': recall,
        'precision': precision,
        'f1': f1_score
    }
    return results


