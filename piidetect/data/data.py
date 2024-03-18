"""
Module: data

A module for managing PII (Personally Identifiable Information) datasets.

Classes:
    DatasetPII: A class for managing PII datasets.
"""

import os
import json
from itertools import chain


class DatasetPII:
    """
    A class for managing PII (Personally Identifiable Information) datasets.

    Args:
        cross_val (bool): Flag indicating whether the class is used for cross-validation.
        config (object): Configuration object containing paths and other settings.

    Attributes:
        cross_val (bool): Flag indicating whether the class is used for cross-validation.
        config (object): Configuration object containing paths and other settings.
        path_data (str): Path to the main dataset.
        extra_data (list): List of paths to additional datasets.
        all_labels (list): List of all unique labels in the dataset.
        label2id (dict): Mapping of labels to unique identifiers.
        id2label (dict): Mapping of unique identifiers to labels.
        train_data (list): List containing the training data.
        path_folds (str): Path to the directory containing cross-validation folds.
    """

    def __init__(self, cross_val=False, validation=False, config=None):
        self.__cross_val = cross_val
        self.config = config
        self.path_data = config.PATH_DATA
        self.extra_data = config.EXTRA_DATA

        (
            self.all_labels,
            self.label2id,
            self.id2label,
        ) = self._get_label_mapping()

        if cross_val:
            self.path_folds = config.PATH_FOLDS

        if validation:
            self.train_data_path = os.path.join(self.path_data, "holdout", "train.json")
            self.validation_data_path = os.path.join(
                self.path_data, "holdout", "validation.json"
            )
        else:
            self.train_data_path = os.path.join(self.path_data, "train.json")

    def load_train_data(self):
        """
        Load the training data from the main dataset and any additional datasets.

        Returns:
            list: List containing the training data.
        """
        if self.__cross_val:
            raise ValueError("load_train_data method can be used with cross_val=False.")
        with open(self.train_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if self.extra_data:
            with open(
                os.path.join(self.path_data, "external_datasets.json"),
                "r",
                encoding="utf-8",
            ) as file:
                extra_data = json.load(file)
            for external_dataset in self.extra_data:
                data += extra_data[external_dataset]
        return data

    def load_validation_data(self):
        with open(self.validation_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data

    def load_train_splits(self, val_id):
        """
        Load training splits for cross-validation.

        Args:
            val_id (int): Identifier for the validation split.

        Returns:
            list: List containing the training data.
        """
        if not self.__cross_val:
            raise ValueError(
                "load_train_splits method can be used with cross_val=True."
            )
        folds = os.listdir(self.path_folds)
        folds = [fold for fold in folds if fold.endswith("json")]
        train_folds = [
            fold for fold in folds if fold != "fold_" + str(val_id) + ".json"
        ]
        data = []
        for fold in train_folds:
            with open(
                os.path.join(self.path_folds, fold), "r", encoding="utf-8"
            ) as file:
                data += json.load(file)
        if self.extra_data:
            with open(
                os.path.join(self.path_data, "external_datasets.json"),
                "r",
                encoding="utf-8",
            ) as file:
                extra_data = json.load(file)
            for external_dataset in self.extra_data:
                data += extra_data[external_dataset]
        return data

    def load_validation_split(self, val_id):
        """
        Load a validation split for cross-validation.

        Args:
            val_id (int): Identifier for the validation split.
            path_folds (str): Path to the directory containing cross-validation folds.

        Returns:
            list: List containing the validation data.
        """
        if not self.__cross_val:
            raise ValueError(
                "load_validation_split method can be used with cross_val=True."
            )
        fold = "fold_" + str(val_id) + ".json"
        with open(os.path.join(self.path_folds, fold), "r", encoding="utf-8") as file:
            data = json.load(file)
        return data

    def _get_full_data(self):
        """
        Load the full dataset.

        Returns:
            list: List containing the full dataset.
        """
        with open(
            os.path.join(self.path_data, "train.json"), "r", encoding="utf-8"
        ) as file:
            data = json.load(file)
        return data

    def _get_label_mapping(self):
        """
        Extract label mappings from the dataset.

        Returns:
            tuple: A tuple containing:
                - all_labels (list): List of all unique labels in the dataset.
                - label2id (dict): Mapping of labels to unique identifiers.
                - id2label (dict): Mapping of unique identifiers to labels.
        """
        data = self._get_full_data()
        all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
        label2id = {l: i for i, l in enumerate(all_labels)}
        id2label = {v: k for k, v in label2id.items()}
        return all_labels, label2id, id2label
