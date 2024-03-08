from functools import partial

from itertools import chain

from transformers import AutoTokenizer, Trainer
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification


from utils.utils_data import get_full_data, load_train_splits
from utils.utils_train import prepare_dataset
from utils.utils_train import *
from utils.utils_config import *



CONFIG = "../configs/config.yaml"


def train(config):

    set_random_seeds(config.SEED)
    args = get_trainer_args(config)
 
    for i in range(config.FOLDS):
        train_(args, i, config)
        break
  
def train_(args, val_id, config):
    
    data = get_full_data(config.PATH_DATA)
    train_data = load_train_splits(val_id, config.PATH_FOLDS, config.PATH_DATA, config.PATH_NICHOLAS)

    
    all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
    label2id = {l: i for i,l in enumerate(all_labels)}
    id2label = {v:k for k,v in label2id.items()}
    
    tokenizer = AutoTokenizer.from_pretrained(config.TRAINING_MODEL_PATH)

    ds = prepare_dataset(train_data, 
                        tokenizer=tokenizer,
                        all_labels=all_labels,
                        label2id=label2id,
                        id2label=id2label,
                        max_length=config.TRAINING_MAX_LENGTH,
                        seed=config.SEED)

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
            compute_metrics=partial(compute_metrics, all_labels=all_labels),
        )

    trainer.train()

    trainer.save_model("deberta3base_debug")
    tokenizer.save_pretrained("deberta3base_debug")



if __name__=="__main__":
    
    config = get_config(CONFIG)
    train(config)