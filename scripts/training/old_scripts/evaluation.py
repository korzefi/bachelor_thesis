# Filip Korzeniewski

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef


def evaluate(y_true, y_pred):
    """
    Evaluates the performance metrics given true labels and predictions.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return accuracy, precision, recall, f1score, mcc


def list_summary(name, data):
    """
    Prints the summary of unique values and their counts in the data.
    """
    print(name)
    unique, count = np.unique(data, return_counts=True)
    print(dict(zip(unique, count)))
