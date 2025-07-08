# [2024-06-10] Modular Pipeline Configuration & Usage Guide

## 📋 Configuration Summary Table

| Pipeline         | Config File Location                        | Key Options (YAML)                                                                                  | Description                                                      |
|------------------|--------------------------------------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| Data Engineering | `scripts/data_engineering/config.yaml`      | `input_fasta`, `split_dir`, `csv_dir`, `cleaned_dir`, `periods_dir`, `lines_per_file`, `start_line`, `max_files`, `min_len`, `max_len`, `division` | Paths and parameters for splitting, cleaning, and periodizing    |
| Clustering       | `scripts/clustering/config.yaml`            | `input_dir`, `vector_dir`, `cluster_dir`, `centroids_csv`, `linked_csv`, `protvec_path`, `n_clusters` | Paths and clustering parameters                                  |
| Training         | `scripts/training/config.yaml`              | `train_file`, `valid_file`, `test_file`, `model_out`, `model_type`, `batch_size`, `epochs`, `lr`, `attn_seq_length` | Data, model, and training hyperparameters                        |
| Inference        | `scripts/training/inference_config.yaml`    | `model_type`, `model_path`, `input_file`, `output_file`, `attn_seq_length`                          | Model and data for batch inference                               |
| Deployment       | `scripts/deployment/deployment_config.yaml` | `model_path`, `model_type`, `attn_seq_length`, `port`                                               | Model, type, attention, and server port for FastAPI deployment   |

---

## 🛠️ User Documentation & Examples

### 1. Data Engineering Pipeline

**Config file:** `scripts/data_engineering/config.yaml`
```yaml
input_fasta: data/input/spikeprot_shortened.fasta
split_dir: data/processed/split_fasta
csv_dir: data/processed/csv
cleaned_dir: data/processed/cleaned
periods_dir: data/processed/periods
lines_per_file: 100000
start_line: 1
max_files: 50
min_len: 1260
max_len: 1280
division: month  # options: month, quarter, year
```
**Run with config:**
```bash
python scripts/data_engineering/run_data_engineering.py --config scripts/data_engineering/config.yaml
```
**Override a parameter:**
```bash
python scripts/data_engineering/run_data_engineering.py --config scripts/data_engineering/config.yaml --division quarter
```

---

### 2. Clustering Pipeline

**Config file:** `scripts/clustering/config.yaml`
```yaml
input_dir: data/processed/periods
vector_dir: data/processed/vectors
cluster_dir: data/processed/clusters
centroids_csv: data/processed/centroids.csv
linked_csv: data/processed/linked_clusters.csv
protvec_path: data/input/protVec_100d_3grams.csv
n_clusters: 8
```
**Run with config:**
```bash
python scripts/clustering/run_clustering.py --config scripts/clustering/config.yaml
```

---

### 3. Training Pipeline

**Config file:** `scripts/training/config.yaml`
```yaml
train_file: data/processed/train.csv
valid_file: data/processed/valid.csv
test_file: data/processed/test.csv
model_out: data/models/model.pth
model_type: rnn  # options: rnn, attn
batch_size: 256
epochs: 20
lr: 0.001
attn_seq_length: null
```
**Run with config:**
```bash
python scripts/training/train_model.py --config scripts/training/config.yaml
```

---

### 4. Inference Pipeline

**Config file:** `scripts/training/inference_config.yaml`
```yaml
model_type: rnn  # options: rnn, attn
model_path: data/models/model.pth
input_file: data/processed/test.csv
output_file: data/predictions/test_preds.npy
attn_seq_length: null
```
**Run with config:**
```bash
python scripts/training/run_inference.py --config scripts/training/inference_config.yaml
```

---

### 5. Deployment (FastAPI + Docker)

**Config file:** `scripts/deployment/deployment_config.yaml`
```yaml
model_path: /app/model.pth
model_type: rnn  # options: rnn, attn
attn_seq_length: null
port: 8000
```
**Build Docker image:**
```bash
docker build -t spikeprot-model-server -f scripts/deployment/Dockerfile .
```
**Run the server (using config):**
```bash
docker run -p 8000:8000 \
  -e DEPLOYMENT_CONFIG=scripts/deployment/deployment_config.yaml \
  spikeprot-model-server
```
**Make a prediction:**
```json
POST http://localhost:8000/predict
{
  "data": [[...], [...], ...],
  "model_type": "rnn",
  "attn_seq_length": 10
}
```

---

## 📝 Notes

- **All config files are in YAML format** for readability and easy editing.
- **All orchestrator scripts** support config files and CLI overrides.
- **Environment variables** can override deployment config values for Docker/production.
- **Legacy scripts** are preserved in `old_scripts` folders for reference.

If you need **example input data**, **sample config files**, or **step-by-step workflow guides**, see the sections below or contact the maintainers.

---

# [2024-06-10] Training Pipeline Refactor

## New Training Folder Structure

- `scripts/training/`: Modular scripts for data loading, model definitions, training, evaluation, and orchestration.
- `scripts/training/old_scripts/`: All legacy training scripts are archived here for reference.

## Training Pipeline Usage

To run the full training pipeline:

```bash
python scripts/training/run_training.py \
  --model_type rnn|attn \
  --train_file <train_csv> \
  --valid_file <valid_csv> \
  --model_out <model_output_path> \
  [--batch_size 256] [--epochs 20] [--lr 0.001] [--attn_seq_length <int>]
```

Each step can also be run independently:
- `load_data.py`: Load a dataset from CSV.
- `model_definitions.py`: PyTorch model classes (RNN, Attention RNN).
- `train_model.py`: Train a model and save it.
- `evaluate_model.py`: Evaluate predictions using scikit-learn metrics (accuracy, precision, recall, F1, MCC).

## Legacy Scripts

All previous training scripts are now in `scripts/training/old_scripts/` for reference and backward compatibility.

## Simplification

- All metrics and evaluation now use scikit-learn functions.
- Training and batching use PyTorch DataLoader and standard best practices.
- Code is modular, minimal, and easy to extend.

---

# [2024-06-10] Clustering Pipeline Refactor

## New Clustering Folder Structure

- `scripts/clustering/`: Modular scripts for sequence vectorization, clustering, centroid computation, and cluster linking.
- `scripts/clustering/old_scripts/`: All legacy clustering scripts are archived here for reference.

## Clustering Pipeline Usage

To run the full clustering pipeline:

```bash
python scripts/clustering/run_clustering.py \
  --input_dir <periodized_csv_dir> \
  --vector_dir <vectorized_csv_dir> \
  --cluster_dir <clustered_csv_dir> \
  --centroids_csv <centroids_output_csv> \
  --linked_csv <linked_clusters_output_csv> \
  --protvec_path <protvec_embedding_csv> \
  [--n_clusters 8]
```

Each step can also be run independently:
- `vectorize_sequences.py`: Vectorize all sequences in period CSVs using protVec.
- `cluster_sequences.py`: Cluster all vectorized period CSVs using KMeans.
- `compute_centroids.py`: Compute centroids for all clustered period CSVs.
- `link_clusters.py`: Link clusters between consecutive periods based on centroid similarity.

## Legacy Scripts

All previous clustering scripts are now in `scripts/clustering/old_scripts/` for reference and backward compatibility.

---

# Bachelor thesis: Prediction of amino acids mutations in SARS-CoV-2 Spike protein using recurrent neural networks

### Additional tools:

#### FastTree - tool for generating phylogenetic tree

`./FastTree -gamma -lg -wag -boot 100 -sprlength 1000 -log fasttree.log spikeprot_batch_data-5002001-5003001.fasta > processed.tree`

#### multiple sequence alignment

`mafft --auto --amino  spikeprot_batch_data-5000001-5010001.fasta > test2`

## SCRIPTS

All scripts are placed in `/scripts` directory
1. `/scripts/preprocessing` contain scripts considering data preprocessing in case of creating datasets
2. `/scripts/clustering` contain scripts strictly helping clustering part
3. `/scripts/training` contain scripts related to traning models with previously prepared datasets

### Preprocessing and clustering

The `scripts/preprocessing/config` and `scripts/clustering/config` files should be filled as needed 

1. `grouping_raw_data.py` is the script converting raw `.fasta` files into cleaned, filtered and divided by periods `.csv` files
2. To find number of clusters for all periods, the `making_clusters.py` script should be used. The periods we want the script to operate on should be selected in the top of the file – `FILENAME_TO_BE_PROCEED`
It is possible to switch between 2 modes:\
a) clustering with particular `k` – figure for clustering with this `k` may be produced: `USE_RANGE_CLUSTERS=False` and `N_CLUSTERS=<k>` \
b) clustering using whole range of clusters – elbow method figure may be produced: `USE_RANGE_CLUSTERS=False` and in config `INIT_N_CLUSTERS` and `END_N_CLUSTERS` set respectively as left limit and right limit
3. After finding appropriate values of `k`, `ClusterToProceed` in config file should be filled appropriately
4. To create create clusters with proper `k`s, link them and create final datasets `clustering.py` script should be used.
5. Dataset should be split in a way user wants for train and test datasets (should be done manually, for example using `pandas` command)

In case of creating final datasets not at once (time consuming process) but at multiple tries, one should modify config and `EpitopeDataCreator.create_final_data()`
Also, the `DatasetRefiller` should be used instead of vanilla `EpitopeDataCreator`

### Training

The `scripts/training/config` file should be filled as needed 

The whole process is done in `training.py`. However to select which model should be used, one is supposed to uncomment the needed part in lines 52-79
The hyperparameters are set separately for each model – the models code is placed in `models.py`


## Additional information

### Clustering 

Clustering is done by creating clusters for each period and then linking those clusters to themselves. Clusters are
linked in a way that euclidean distance is taken in concesutive years - if cluster from period i has the minimum
distance to other cluster from period i+1, it is linked. Then for each sequence in those clusters, data is constructed (
window = 10 periods by default => means 10 columns of sequences and y column is for window+1 period).

Clustering is done using K-means and to visualization - PCA and t-sne reduction methods.

### Creating final datasets

- add cluster column for files in periods/unique representing num of cluster it belongs to (indexing from 0) - for this
  case data need to be clustered again (do not need to be visualized) and the corresponding labels have to be added as a
  column num to these files
- sort the centroids data by period
- clusters from consecutive periods are needed to be linked on the base of centroids distances and indexes registered (
  ex. [0, 0, 1, 3, 1, 1, 1, 2, 0, 0])
- pick randomly n of rows from a 'linked' cluster and get current position (+/- 2) positions
- after creating given number of data, make it unique and refill with missing number of data
- translate the data into embedded data vectors - window_size number of column, in each column triplet of
  numbers [num1, num2, num3]

1 row taken from clusters gives 317 positions data

### epitopes_similarity_threshold parameter

It is good to do double filtration when creating final dataset. There are sequences taken randomly
accordingly to the clusters created, but it can be noticed that some sequences are totally different - have all epitopes
mutated. **This may indicate that they were not exactly properly clustered.** There should be set  
**epitopes_similarity_threshold** standing for the % of mutated values relating to the previous sequence. If the threshold is passed, for the sequence,
the sequence is neglected then and new sequence is picked.

##### Example

We have 2 consecutive sequences (from consecutive periods) that are picked to be linked and the threshold 50%. If the
sequence from period *i+1* has different values for over 50% of all epitopes than the sequence from period *i*, new
sequence (from period *i+1*) should be chosen and checked with the same criterion.
