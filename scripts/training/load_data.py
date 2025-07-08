import pandas as pd
import numpy as np
import argparse
import torch

def load_dataset(filepath, label_col='y'):
    df = pd.read_csv(filepath)
    labels = df[label_col].values
    features = df.drop(columns=[label_col]).values
    return features, labels

def main():
    parser = argparse.ArgumentParser(description='Load a dataset and print its shape.')
    parser.add_argument('--file', required=True, help='Path to CSV file')
    parser.add_argument('--label_col', default='y', help='Name of the label column')
    args = parser.parse_args()
    X, y = load_dataset(args.file, args.label_col)
    print(f'Features shape: {X.shape}')
    print(f'Labels shape: {y.shape}')

if __name__ == '__main__':
    main()
