import pandas as pd
import sys

sys.path.append("../..")  # Add parent directory of 'src' to the Python path

from piidetect.metric import metric


df_gt = pd.read_csv("good_model/gt.csv")
df_pred = pd.read_csv("preds.csv")

x = metric.compute_metrics_eval(df_pred, df_gt)
print(x)
