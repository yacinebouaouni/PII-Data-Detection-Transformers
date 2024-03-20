"""
Module: metric

This module provides functions for computing evaluation metrics.
"""

import pandas as pd
from collections import defaultdict
from typing import Dict


class Evaluator:
    """
    A class for evaluating performance metrics.
    """

    @staticmethod
    def compute_confusion_matrix_alpha(pred_df: pd.DataFrame, gt_df: pd.DataFrame):
        """
        Compute confusion matrix alpha.

        Parameters:
        - pred_df (DataFrame): DataFrame containing predicted PII labels.
        - gt_df (DataFrame): DataFrame containing ground truth PII labels.

        Returns:
        - dict: Dictionary containing FP, FN, and TP counts.
        """
        df = pred_df.merge(
            gt_df, how="outer", on=["document", "token"], suffixes=("_pred", "_gt")
        )
        df["cm"] = ""

        df.loc[df.label_gt.isna(), "cm"] = "FP"
        df.loc[df.label_pred.isna(), "cm"] = "FN"
        df.loc[(df.label_gt.notna()) & (df.label_gt != df.label_pred), "cm"] = "FN"
        df.loc[
            (df.label_pred.notna())
            & (df.label_gt.notna())
            & (df.label_gt == df.label_pred),
            "cm",
        ] = "TP"

        fp = (df["cm"] == "FP").sum()
        fn = (df["cm"] == "FN").sum()
        tp = (df["cm"] == "TP").sum()

        return {"fp": fp, "fn": fn, "tp": tp}

    @staticmethod
    def compute_metrics_eval(df_pred, df_gt, beta=5, log=True):
        """
        Compute evaluation metrics.

        Parameters:
        - df_pred (DataFrame): DataFrame containing predicted PII labels.
        - df_gt (DataFrame): DataFrame containing ground truth PII labels.
        - beta (float): The beta parameter for the F-beta score
        - log (bool): Whether to log the computed metrics.

        Returns:
        - tuple: Tuple containing precision, recall, and f1 score.
        """
        # Compute precision, recall, and F1 score
        confusion_matrix = Evaluator.compute_confusion_matrix_alpha(df_pred, df_gt)
        tp = confusion_matrix["tp"]
        fp = confusion_matrix["fp"]
        fn = confusion_matrix["fn"]

        f1 = (
            (1 + (beta**2))
            * tp
            / (((1 + (beta**2)) * tp) + ((beta**2) * fn) + fp)
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if log:
            print("True Positives:", tp)
            print("False Positives:", fp)
            print("False Negatives:", fn)
            print("Precision:", precision)
            print("Recall:", recall)
            print("f1 competition:", f1)

        return precision, recall, f1


class PRFScore:
    """A precision / recall / F score."""

    def __init__(
        self,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
    ) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):  # in-place add
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * ((p * r) / (p + r + 1e-100))

    @property
    def f5(self) -> float:
        beta = 5
        p = self.precision
        r = self.recall

        fbeta = (1 + (beta**2)) * p * r / ((beta**2) * p + r + 1e-100)
        return fbeta

    def to_dict(self) -> Dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f5": self.f5}


def compute_metrics(pred_df, gt_df):
    """
    Compute the LB metric (lb) and other auxiliary metrics
    """

    references = {(row.document, row.token, row.label) for row in gt_df.itertuples()}
    predictions = {(row.document, row.token, row.label) for row in pred_df.itertuples()}

    score_per_type = defaultdict(PRFScore)
    references = set(references)

    for ex in predictions:
        pred_type = ex[-1]  # (document, token, label)
        if pred_type != "O":
            pred_type = pred_type[2:]  # avoid B- and I- prefix

        if pred_type not in score_per_type:
            score_per_type[pred_type] = PRFScore()

        if ex in references:
            score_per_type[pred_type].tp += 1
            references.remove(ex)
        else:
            score_per_type[pred_type].fp += 1

    for doc, tok, ref_type in references:
        if ref_type != "O":
            ref_type = ref_type[2:]  # avoid B- and I- prefix

        if ref_type not in score_per_type:
            score_per_type[ref_type] = PRFScore()
        score_per_type[ref_type].fn += 1

    totals = PRFScore()

    for prf in score_per_type.values():
        totals += prf

    return {
        "precision": totals.precision,
        "recall": totals.recall,
        "f5": totals.f5,
        "score_per_entity": {
            k: v.to_dict() for k, v in score_per_type.items() if k != "O"
        },
    }
