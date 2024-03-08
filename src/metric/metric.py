import pandas as pd

def compute_confusion_matrix_alpha(pred_df, gt_df):
    """
    Parameters:
    - pred_df (DataFrame): DataFrame containing predicted PII labels.
    - gt_df (DataFrame): DataFrame containing ground truth PII labels.
    - beta (float): The beta parameter for the F-beta score, controlling the trade-off between precision and recall.

    Returns:
    - float: Micro F-beta score.
    """   
    df = pred_df.merge(gt_df,how='outer',on=['document',"token"],suffixes=('_pred','_gt'))

    df['cm'] = ""

    df.loc[df.label_gt.isna(),'cm'] = "FP"


    df.loc[df.label_pred.isna(),'cm'] = "FN"
    df.loc[(df.label_gt.notna()) & (df.label_gt!=df.label_pred),'cm'] = "FN"

    df.loc[(df.label_pred.notna()) & (df.label_gt.notna()) & (df.label_gt==df.label_pred),'cm'] = "TP"
    
    FP = (df['cm']=="FP").sum()
    FN = (df['cm']=="FN").sum()
    TP = (df['cm']=="TP").sum()

    return {"fp":FP, "fn": FN, "tp":TP}


def compute_confusion_matrix_beta(predictions_df, groundtruth_df):
    # Initialize counters
    TP = 0
    FN = 0
    FP = 0

    # Iterate through each (document, token) pair in groundtruth
    for index, row in groundtruth_df.iterrows():
        doc = row['document']
        token = row['token']
        label = row['label']

        # Check if the pair exists in predictions
        if ((predictions_df['document'] == doc) & (predictions_df['token'] == token)).any():
            # Get the label from predictions
            predicted_label = predictions_df[(predictions_df['document'] == doc) & (predictions_df['token'] == token)]['label'].iloc[0]

            # Compare labels
            if predicted_label == label:
                TP += 1  # True positive
            else:
                FN += 1  # False negative
        else:
            FN += 1  # False negative

    # Iterate through each (document, token) pair in predictions
    for _, row in predictions_df.iterrows():
        doc = row['document']
        token = row['token']

        # Check if the pair exists in groundtruth
        if not ((groundtruth_df['document'] == doc) & (groundtruth_df['token'] == token)).any():
            FP += 1  # False positive

    return {"fp":FP, "fn": FN, "tp":TP}


def compute_metrics_eval(df_pred, df_gt, beta=5, log=True):
    # Compute precision and recall
    confusion_matrix = compute_confusion_matrix_alpha(df_pred, df_gt)
    tp = confusion_matrix['tp']
    fp = confusion_matrix['fp']
    fn = confusion_matrix['fn']
    
    f1 = (1+(beta**2))*tp/(((1+(beta**2))*tp) + ((beta**2)*fn) + fp)
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


