import logging

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scripts.clustering.config import MakingClusters, TsneConfig


class ReductionMethod:
    def __init__(self):
        self.data_vecs = []
        self.labels = []

    def reduce(self, clusters):
        logging.info('Clusters data are being reduced...')
        self.data_vecs = clusters['data']
        self.labels = clusters['labels']
        reduced_data = self._execute()
        for i in range(len(reduced_data)):
            colors = 10 * ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6',
                           '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
                           '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#000000']
            plt.plot(reduced_data[i][0], reduced_data[i][1], color=colors[self.labels[i]], marker='.', markersize=10)

    def _execute(self):
        return []


class TSNEMethod(ReductionMethod):
    def _execute(self):
        all_jobs_activated = -1
        tsne = TSNE(init='pca',
                    n_components=MakingClusters.PLOT_CLUSTERS_DIMS,
                    n_jobs=all_jobs_activated,
                    learning_rate='auto',
                    n_iter=TsneConfig.N_ITER,
                    n_iter_without_progress=TsneConfig.N_ITER_WITHOUT_PROGRESS)
        return tsne.fit_transform(self.data_vecs)


class PCAMethod(ReductionMethod):
    def _execute(self):
        pca = PCA(n_components=MakingClusters.PLOT_CLUSTERS_DIMS)
        reduced_data = pca.fit_transform(self.data_vecs)
        logging.info(f'Explained variance:{pca.explained_variance_ratio_}')
        return reduced_data


class ReductionMethodFactory:
    def __init__(self):
        self.methods = {'TSNE': TSNEMethod,
                        'PCA': PCAMethod}

    def create_method(self, method_name):
        return self.methods[method_name]()


def plot_clusters(clusters, method: ReductionMethod, filename, save_fig=True):
    method.reduce(clusters)
    if save_fig:
        plt.savefig(f'{MakingClusters.CLUSTER_PLOT_PATH}/{filename}')
    plt.show()


def plot_elbow_method(all_clusters: {}, save_fig=False):
    if MakingClusters.CLUSTER_METHOD != 'KMeans':
        logging.warning(f'Elbow cannot be plotted for {MakingClusters.CLUSTER_METHOD}, change to KMeans')
    for clusters in all_clusters:
        plt.plot(clusters['n_clusters'], clusters['distortion'], 'k.', markersize=10)
    plt.show()
