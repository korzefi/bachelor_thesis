import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model_definitions import RnnModel, AttnRnnModel
from load_data import load_dataset
import yaml

def train_model(model_type, train_file, valid_file, model_out, batch_size=256, epochs=20, lr=0.001, attn_seq_length=None):
    X_train, y_train = load_dataset(train_file)
    X_valid, y_valid = load_dataset(valid_file)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=batch_size)

    input_dim = X_train.shape[1] if len(X_train.shape) == 2 else X_train.shape[2]
    if model_type == 'rnn':
        model = RnnModel(input_dim=input_dim)
    elif model_type == 'attn':
        if attn_seq_length is None:
            attn_seq_length = X_train.shape[1]
        model = AttnRnnModel(input_dim=input_dim, seq_length=attn_seq_length)
    else:
        raise ValueError('Unknown model type')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch) if model_type == 'rnn' else model(X_batch)[0]
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')

        # Optionally, add validation metrics here

    torch.save(model.state_dict(), model_out)
    print(f'Model saved to {model_out}')


def main():
    parser = argparse.ArgumentParser(description='Train a model on a dataset.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file (optional)')
    parser.add_argument('--model_type', choices=['rnn', 'attn'], default=None)
    parser.add_argument('--train_file', default=None)
    parser.add_argument('--valid_file', default=None)
    parser.add_argument('--model_out', default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--attn_seq_length', type=int, default=None)
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    def get_param(key, default=None):
        return getattr(args, key) if getattr(args, key) not in [None, 'None'] else config.get(key, default)

    train_model(
        model_type=get_param('model_type', 'rnn'),
        train_file=get_param('train_file'),
        valid_file=get_param('valid_file'),
        model_out=get_param('model_out', 'model.pth'),
        batch_size=int(get_param('batch_size', 256)),
        epochs=int(get_param('epochs', 20)),
        lr=float(get_param('lr', 0.001)),
        attn_seq_length=None if get_param('attn_seq_length') in [None, 'null', 'None'] else int(get_param('attn_seq_length'))
    )

if __name__ == '__main__':
    main()
