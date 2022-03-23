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

Clustering is done by creating clusters for each period and then linking those clusters to themselves.
Clusters are linked in a way that euclidean distance is taken in concesutive years - if cluster from period i has the minimum distance to other cluster from period i+1, it is linked.
Then for each sequence in those clusters, data is constructed (window = 10 periods by default => means 10 columns of sequences and y column is for window+1 period).

Clustering is done using K-means and to visualization - PCA and t-sne reduction methods.

