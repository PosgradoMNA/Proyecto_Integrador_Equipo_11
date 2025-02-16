import pandas as pd
from sklearn.metrics import (classification_report, roc_auc_score, precision_recall_curve,
                             auc, roc_curve, det_curve)
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    """
    Carga los datos preprocesados del análisis exploratorio

    Usage:
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    """
    X_train = pd.read_parquet('../data/preprocessed/X_train.parquet')
    y_train = pd.read_parquet('../data/preprocessed/y_train.parquet')
    X_val = pd.read_parquet('../data/preprocessed/X_val.parquet')
    y_val = pd.read_parquet('../data/preprocessed/y_val.parquet')
    X_test = pd.read_parquet('../data/preprocessed/X_test.parquet')
    y_test = pd.read_parquet('../data/preprocessed/y_test.parquet')


    assert X_train.shape[0] == y_train.shape[0], "Dimensions don't match"
    assert X_val.shape[0] == y_val.shape[0], "Dimensions, don't match"

    features_x_train = pd.DataFrame(X_train['caracteristica_0'].to_list())
    features_y_train = pd.DataFrame(X_train['caracteristica_1'].to_list())
    all_features_train = pd.concat([features_x_train, features_y_train], axis=1)
    all_features_train.columns = [f"feature_{i}" for i in range(all_features_train.shape[1])]

    features_x_val = pd.DataFrame(X_val['caracteristica_0'].to_list())
    features_y_val = pd.DataFrame(X_val['caracteristica_1'].to_list())
    all_features_val = pd.concat([features_x_val, features_y_val], axis=1)
    all_features_val.columns = [f"feature_{i}" for i in range(all_features_val.shape[1])]

    return all_features_train, y_train, all_features_val, y_val, X_test, y_test


def print_classification_report(y_true, y_pred, y_prob, label=""):
    print(f"{label} Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    roc_auc = roc_auc_score(y_true, y_prob)
    print(f"{label} ROC AUC: {roc_auc:.4f}")

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    print(f"{label} Precision-Recall AUC: {pr_auc:.4f}")


def plot_roc_det_curve(model_name, y_val, y_pred):
  plt.figure(figsize=(28, 14))
  palette = sns.color_palette("husl")

  fpr, tpr, _ = roc_curve(y_val, y_pred)
  fpr_det, fnr_det, _ = det_curve(y_val, y_pred)

  plt.subplot(1, 2, 1)
  sns.lineplot(x=fpr, y=tpr, label=f"{model_name} (AUC = {roc_auc_score(y_val, y_pred):.2f})",
                  color=palette[0])

  plt.subplot(1, 2, 2)
  sns.lineplot(x=fpr_det, y=fnr_det, label=model_name, color=palette[0])

  plt.subplot(1, 2, 1)
  plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("ROC Curve Comparison")
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.xlabel("False Positive Rate")
  plt.ylabel("False Negative Rate")
  plt.title("DET Curve Comparison")
  plt.legend()

  plt.tight_layout()
  plt.show()


# def plot_roc_det_curve(model_name, y_val, y_pred_proba, label_name):
#   plt.figure(figsize=(28, 14))
#   palette = sns.color_palette("husl")

#   fpr, tpr, _ = roc_curve(y_val[label_name], y_pred_proba)
#   fpr_det, fnr_det, _ = det_curve(y_val[label_name], y_pred_proba)

#   plt.subplot(1, 2, 1)
#   sns.lineplot(x=fpr, y=tpr, label=f"{model_name} (AUC = {roc_auc_score(y_val[label_name], y_pred_proba):.2f})",
#                   color=palette[0])

#   plt.subplot(1, 2, 2)
#   sns.lineplot(x=fpr_det, y=fnr_det, label=model_name, color=palette[0])

#   plt.subplot(1, 2, 1)
#   plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
#   plt.xlabel("False Positive Rate")
#   plt.ylabel("True Positive Rate")
#   plt.title("ROC Curve Comparison")
#   plt.legend()

#   plt.subplot(1, 2, 2)
#   plt.xlabel("False Positive Rate")
#   plt.ylabel("False Negative Rate")
#   plt.title("DET Curve Comparison")
#   plt.legend()

#   plt.tight_layout()
#   plt.show()