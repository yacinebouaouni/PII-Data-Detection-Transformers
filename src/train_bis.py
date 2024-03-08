
from transformers import AutoTokenizer, Trainer
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

from utils.utils_config import *
from utils.utils_data import *
from utils.utils_test import *
from utils.utils_train import *

from functools import partial



CONFIG = "../configs/config.yaml"


def train(config):

    set_random_seeds(config.SEED)
    args = get_trainer_args(config)
 
    for i in range(config.FOLDS):
        train_(i, args, config)
        break
  

def train_(val_id, args, config):
    train_data = load_train_splits(val_id, config.PATH_FOLDS, config.PATH_DATA, config.PATH_NICHOLAS)
    validation_data = load_validation_split(val_id, config.PATH_FOLDS)
    data = get_full_data(config.PATH_DATA)
    all_labels, label2id, id2label = get_label_mapping(data)
    tokenizer = AutoTokenizer.from_pretrained(config.TRAINING_MODEL_PATH)

    ds_train = prepare_dataset(train_data, tokenizer, all_labels, label2id, id2label, config.TRAINING_MAX_LENGTH, config.SEED)
    ds_validation = prepare_dataset(validation_data, tokenizer, all_labels, label2id, id2label, config.TRAINING_MAX_LENGTH, config.SEED)

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
            eval_dataset=ds_validation, 
            data_collator=collator, 
            tokenizer=tokenizer,
            compute_metrics=partial(compute_metrics, all_labels=all_labels),
        )
    
    

    print(f'start training for evaluation fold = {val_id}')
    training_output = trainer.train()
    results = trainer.evaluate()
    print(results)
    if config.SAVE_MODELS:
       trainer.save_model(config.MODEL_SAVE)
       tokenizer.save_pretrained(config.MODEL_SAVE)
       

    

    
if __name__ == "__main__":
    
    config = get_config(CONFIG)
        
    for EPOCHS in [1]:
        for BATCH in [12]:
            for ACCUMULATION in [1]:
                config.EPOCHS = EPOCHS
                config.BATCH  = BATCH
                config.ACCUMULATION = ACCUMULATION 
                print(f'Training for Epochs = {config.EPOCHS}, BATCH={config.BATCH}, ACCUMULATION={config.ACCUMULATION}')
                train(config)
    