"""
Module: metric

This module provides functions for computing evaluation metrics.
"""

import pandas as pd

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
        df = pred_df.merge(gt_df, how='outer', on=['document', 'token'], suffixes=('_pred', '_gt'))
        df['cm'] = ""

        df.loc[df.label_gt.isna(), 'cm'] = "FP"
        df.loc[df.label_pred.isna(), 'cm'] = "FN"
        df.loc[(df.label_gt.notna()) & (df.label_gt != df.label_pred), 'cm'] = "FN"
        df.loc[(df.label_pred.notna()) & (df.label_gt.notna()) & (df.label_gt == df.label_pred), 'cm'] = "TP"

        fp = (df['cm'] == "FP").sum()
        fn = (df['cm'] == "FN").sum()
        tp = (df['cm'] == "TP").sum()

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
        tp = confusion_matrix['tp']
        fp = confusion_matrix['fp']
        fn = confusion_matrix['fn']

        f1 = (1 + (beta ** 2)) * tp / (((1 + (beta ** 2)) * tp) + ((beta ** 2) * fn) + fp)
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
