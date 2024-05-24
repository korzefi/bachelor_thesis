# Filip Korzeniewski
# Partially adapted from :
# Yin R, Luusua E, Dabrowski J, Zhang Y, Kwoh CK.
# Tempel: time-series mutation prediction of influenza A viruses via attention-based recurrent neural networks.
# Bioinformatics. 2020 May 1;36(9):2697-2704. doi: 10.1093/bioinformatics/btaa050. PMID: 31999330.

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef


def evaluate(y_true, y_pred):
    """
    Evaluates the performance metrics given true labels and predictions.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, f1score, mcc, accuracy


def list_summary(name, data):
    """
    Prints the summary of unique values and their counts in the data.
    """
    print(name)
    unique, count = np.unique(data, return_counts=True)
    print(dict(zip(unique, count)))
