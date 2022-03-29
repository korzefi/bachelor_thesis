# Bachelor thesis: Prediction of amino acids mutations in SARS-CoV-2 Spike protein using recurrent neural networks

## DATA PREPROCESSING

### FastTree - tool for generating phylogenetic tree

`./FastTree -gamma -lg -wag -boot 100 -sprlength 1000 -log fasttree.log spikeprot_batch_data-5002001-5003001.fasta > processed.tree`

### multiple sequence alignment

`mafft --auto --amino  spikeprot_batch_data-5000001-5010001.fasta > test2`

### SCRIPTS

Script that cleans, groups and transform to csv files the data in `.fasta` format.\
All configuration is included in the file at the top level, and more specific configuration in each class.
path: `scripts/utils/grouping_raw_data.py`

### CLUSTERING

Clustering is done by creating clusters for each period and then linking those clusters to themselves. Clusters are
linked in a way that euclidean distance is taken in concesutive years - if cluster from period i has the minimum
distance to other cluster from period i+1, it is linked. Then for each sequence in those clusters, data is constructed (
window = 10 periods by default => means 10 columns of sequences and y column is for window+1 period).

Clustering is done using K-means and to visualization - PCA and t-sne reduction methods.

### TODO

- add cluster column for files in periods/unique representing num of cluster it belongs to (indexing from 0) - for this
  case data need to be clustered again (do not need to be visualized) and the corresponding labels have to be added as a
  column num to these files
- clusters from consecutive periods are needed to be linked on the base of centroids distances and indexes registered (
  ex. [0, 0, 1, 3, 1, 1, 1, 2, 0, 0])
- pick randomly n of rows from a 'linked' cluster and get current position (+/- 2) positions
- after creating given number of data, make it unique and refill with missing number of data
- translate the data into embedded data vectors - window_size number of column, in each column triplet of
  numbers [num1, num2, num3]

1 row taken from clusters gives 317 positions data, only 1000 unique rows must be taken then
