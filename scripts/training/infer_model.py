import argparse
import torch
import numpy as np
from model_definitions import RnnModel, AttnRnnModel
from load_data import load_dataset

def infer(model_type, model_path, input_file, output_file, attn_seq_length=None):
    X, _ = load_dataset(input_file)
    X = torch.tensor(X, dtype=torch.float32)
    input_dim = X.shape[1] if len(X.shape) == 2 else X.shape[2]
    if model_type == 'rnn':
        model = RnnModel(input_dim=input_dim)
    elif model_type == 'attn':
        if attn_seq_length is None:
            attn_seq_length = X.shape[1]
        model = AttnRnnModel(input_dim=input_dim, seq_length=attn_seq_length)
    else:
        raise ValueError('Unknown model type')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        outputs = model(X) if model_type == 'rnn' else model(X)[0]
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    np.save(output_file, preds)
    print(f'Predictions saved to {output_file}')

def main():
    parser = argparse.ArgumentParser(description='Run inference with a trained model.')
    parser.add_argument('--model_type', choices=['rnn', 'attn'], required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--attn_seq_length', type=int, default=None)
    args = parser.parse_args()
    infer(args.model_type, args.model_path, args.input_file, args.output_file, args.attn_seq_length)

if __name__ == '__main__':
    main()
