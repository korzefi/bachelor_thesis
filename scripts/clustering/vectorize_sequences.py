import os
import pandas as pd
import logging
import argparse
from natsort import natsorted

def vectorize_sequences(input_dir, output_dir, protvec_path):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    files = natsorted(files)
    prot_vec = pd.read_csv(protvec_path)
    for file in files:
        input_path = os.path.join(input_dir, file)
        df = pd.read_csv(input_path)
        seqs = df['sequence']
        vectors = []
        for seq in seqs:
            triplets = [seq[i:i+3] for i in range(len(seq)-2)]
            seq_vec = prot_vec[prot_vec['words'].isin(triplets)].drop('words', axis=1).sum().values
            vectors.append(seq_vec)
        vec_df = pd.DataFrame(vectors)
        out_path = os.path.join(output_dir, file)
        vec_df.to_csv(out_path, index=False)
        logging.info(f'Vectorized {file} to {out_path}')
    logging.info(f'Finished vectorizing all sequences in {input_dir} to {output_dir}')

def main():
    parser = argparse.ArgumentParser(description="Vectorize all sequences in period CSVs using protVec embeddings.")
    parser.add_argument('--input_dir', required=True, help='Directory with period CSVs (with sequences)')
    parser.add_argument('--output_dir', required=True, help='Directory to save vectorized CSVs')
    parser.add_argument('--protvec_path', required=True, help='Path to protVec embedding CSV')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    vectorize_sequences(args.input_dir, args.output_dir, args.protvec_path)

if __name__ == '__main__':
    main()
