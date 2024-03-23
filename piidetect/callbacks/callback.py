from utils.utils_test import test_beta
from transformers import TrainerCallback


class EvalCallback(TrainerCallback):
    def __init__(
        self,
        stride,
        threshold,
        path_folds,
        ignored_labels,
        id2label,
        val_id,
        ds_validation,
        device,
    ):
        self.stride = stride
        self.threshold = threshold
        self.ignored_labels = ignored_labels
        self.val_id = val_id
        self.id2label = id2label
        self.path_folds = path_folds
        self.ds_validation = ds_validation
        self.device = device
        self.results_list = []

    def on_epoch_end(self, args, state, control, model, **kwargs):
        # Put model in evaluation mode
        model.eval()
        results = test_beta(
            model.to(self.device),
            self.ds_validation,
            self.stride,
            self.threshold,
            self.path_folds,
            self.ignored_labels,
            self.id2label,
            self.val_id,
            self.device,
        )
        model.train()
        self.results_list.append(results)

    def get_results(self):
        return self.results_list
