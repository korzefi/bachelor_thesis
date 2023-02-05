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
