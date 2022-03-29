import logging

from scripts.clustering.config import LinkingClusters as cfg


def get_epitopes_list() -> []:
    epitopes = [list(range(start, end + 1)) for start, end in cfg.epitopes]
    flat_epitopes = [positions for epitope in epitopes for positions in epitope]
    logging.info(f'All epitopes positions to be checked:\n{flat_epitopes}')
    return flat_epitopes


class ClusterLinker:
    pass
