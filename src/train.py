from transformers import AutoTokenizer, Trainer
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

from utils.utils_data import get_full_data, load_train_splits, get_label_mapping, load_validation_split
from utils.utils_train import *
from utils.utils_config import *
from utils.utils_test import *

import mlflow
from datetime import datetime
import statistics 

CONFIG = "../configs/config.yaml"
EXPERIMENTS_PATH = "../experiments-EPOCH-BATCH-ACC"

def train(config):

    set_random_seeds(config.SEED)
    args = get_trainer_args(config)
    
    current_datetime = datetime.now()
    run_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    # setup mlflow uri and run
    mlflow.set_tracking_uri(EXPERIMENTS_PATH)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(config.to_dict())
    
    f1_list = []
    for i in range(config.FOLDS):
        metrics = train_(args, i, config)
        #log CV metrics and end run
        mlflow.log_metric(f"f1_fold_{i}", metrics['f1'])
        mlflow.log_metric(f"precision_fold_{i}", metrics['precision'])
        mlflow.log_metric(f"recall_fold_{i}", metrics['recall'])
        f1_list.append(metrics['f1'])
        
    f1_mean = sum(f1_list)/len(f1_list)
    mlflow.log_metric("std_f1", f1_mean)
    mlflow.log_metric("std_f1", statistics.pstdev(f1_list))
    
    mlflow.end_run()
  
  
def train_(args, val_id, config):
    
    data = get_full_data(config.PATH_DATA)
    train_data = load_train_splits(val_id, config.PATH_FOLDS, config.PATH_DATA, config.PATH_NICHOLAS)
    validation_data = load_validation_split(val_id, config.PATH_FOLDS)

    all_labels, label2id, id2label = get_label_mapping(data)

    tokenizer = AutoTokenizer.from_pretrained(config.TRAINING_MODEL_PATH)

    ds = prepare_dataset(train_data, 
                        tokenizer=tokenizer,
                        all_labels=all_labels,
                        label2id=label2id,
                        id2label=id2label,
                        max_length=config.TRAINING_MAX_LENGTH,
                        seed=config.SEED)

    ds_validation = prepare_dataset_test(validation_data,
                                         tokenizer=tokenizer,
                                         max_length=config.TRAINING_MAX_LENGTH,
                                         stride=config.STRIDE,
                                         )
    
    
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
            train_dataset=ds, 
            data_collator=collator, 
            tokenizer=tokenizer,
            #compute_metrics=partial(compute_metrics, all_labels=all_labels),
        )

    trainer.train()

    # evaluate model on spacy
    ignored_labels = ('O')
    precision, recall, f1 = test(trainer, ds_validation, config.STRIDE, config.THRESHOLD, config.PATH_FOLDS, ignored_labels, id2label, val_id)
    
    if config.SAVE_MODELS:
        trainer.save_model("deberta3base_debug")
        tokenizer.save_pretrained("deberta3base_debug")
        
    return {"precision":precision, "recall":recall, "f1":f1}



if __name__=="__main__":
    
    config = get_config(CONFIG)
    for EPOCHS in [2,3]:
        for BATCH in [4,6,8]:
            for ACCUMULATION in [1,2]:
                config.EPOCHS = EPOCHS
                config.BATCH  = BATCH
                config.ACCUMULATION = ACCUMULATION 
                print(f'Training for Epochs = {config.EPOCHS}, BATCH={config.BATCH}, ACCUMULATION={config.ACCUMULATION}')
                train(config)
