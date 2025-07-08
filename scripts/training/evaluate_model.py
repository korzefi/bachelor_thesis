import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import argparse

def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    return accuracy, precision, recall, f1score, mcc

def main():
    parser = argparse.ArgumentParser(description='Evaluate predictions using sklearn metrics.')
    parser.add_argument('--y_true', required=True, help='Path to true labels (npy file)')
    parser.add_argument('--y_pred', required=True, help='Path to predicted labels (npy file)')
    args = parser.parse_args()
    y_true = np.load(args.y_true)
    y_pred = np.load(args.y_pred)
    acc, prec, rec, f1, mcc = evaluate(y_true, y_pred)
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'MCC: {mcc:.4f}')

if __name__ == '__main__':
    main()

