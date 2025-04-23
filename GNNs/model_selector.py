import torch
import torch_geometric
import os

from GNNs.GNNs import GCN3PoolNorm, GCN3PoolFun, GCN1PoolNorm, GCN1Pool
from GNNs.MLPs import MLP3PoolFun, MLP3PoolNorm, MLP1PoolFun, MLP1PoolNorm
'''
'''
'''
def string_to_model(paper, dataset, layers):
    """
    Given a paper and a dataset return the cooresponding neural model needed for training.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :returns: torch.nn.module models
    """
    if paper == "GNN":
        if dataset in ['bashapes']:
            return GNN_NodeGCN(10, 4)
        elif dataset in ['bacommunity']:
            return GNN_NodeGCN(10, 8)
        elif dataset in ['treecycle']:
            return GNN_NodeGCN(10, 2)
        elif dataset in ['treegrid']:
            return GNN_NodeGCN(10, 2)
        elif dataset.startswith('ba2'):
            return GNN_GraphGCN(10, 2)
        elif dataset.startswith('ba3'):
            return GNN_GraphGCN(10, 3)
        elif dataset.startswith('bax'):
            return GNN_GraphGCN(10, 3)
        elif dataset == "mutag":
            return GNN_GraphGCN(14, 2)
        else:
            raise NotImplementedError
    elif paper == "PG":

        if dataset in ['bashapes']:
            return PG_NodeGCN(10, 4)
        elif dataset in ['bacommunity']:
            return PG_NodeGCN(10, 8)
        elif dataset in ['treecycle']:
            return PG_NodeGCN(10, 2)
        elif dataset in ['treegrid']:
            return PG_NodeGCN(10, 2)
        elif dataset.startswith("ba2"):
            return PG_GraphGCN(10, 2)
        elif dataset.startswith("ba3"):
            return PG_GraphGCN(10, 3)
        elif dataset.startswith("bax"):
            return PG_GraphGCN(10, 3)
        elif dataset == "mutag":
            return PG_GraphGCN(14, 2)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
'''
def get_channels_classes(dataset):
    if dataset in ['ba2', 'neoba2']:
        return 10, 2
    elif dataset.startswith('mutagenicity'):
        return 14, 2
    elif dataset.endswith('mutagenicity'):
        return 14, 2
    elif dataset.startswith('fc'):
        return 14, 2
    elif dataset.startswith('ac'):
        return 14, 2
    elif dataset.startswith('benzene'):
        return 14, 2
    elif dataset.startswith('mnist'):
        return 5, 10
    else:
        raise NotImplementedError
    
def new_model_selector(dataset, layers, hidden_size, types = 'GCN', config = None):
    channels, classes = get_channels_classes(dataset)
    if config['Norm'] in ['LayerNorm', 'BatchNorm', 'InstanceNorm']:
        Norm = getattr(torch_geometric.nn.norm, config['Norm'])
    elif config['Norm'] in ['RMSNorm']:
        Norm = torch.nn.RMSNorm
    else:
        pass
    
    if types == 'GCN':
        return GraphGCN(channels , classes)
    elif types == 'MLP':
        return GraphMLP(channels, classes, layers = layers, hidden_size = hidden_size)
    elif types == 'GCN3PoolNorm':
        return GCN3PoolNorm(channels, classes, layers = layers, hidden_size = hidden_size, Norm = Norm)
    elif types == 'GCN3PoolFun':
        return GCN3PoolFun(channels, classes, layers = layers, hidden_size = hidden_size, p = config.get('p', 2), norm_dim = config.get('dim', 1))
    elif types == 'GCN1PoolNorm':
        return GCN1PoolNorm(channels, classes, layers = layers, hidden_size = hidden_size, Pool = getattr(torch_geometric.nn.pool, config.get('Pool', 'global_max_pool')), Norm = Norm)
    elif types == 'GCN1Pool':
        return GCN1Pool(channels, classes, layers = layers, hidden_size = hidden_size, Pool = getattr(torch_geometric.nn.pool, config.get('Pool', 'global_max_pool')), Norm = Norm)
    elif types == 'MLP3PoolFun':
        return MLP3PoolFun(channels, classes, layers = layers, hidden_size = hidden_size, p = config.get('p', 2), norm_dim = config.get('dim', 1))
    elif types == 'MLP3PoolNorm':
        return MLP3PoolNorm(channels, classes, layers = layers, hidden_size = hidden_size, Norm = Norm)
    elif types == 'MLP1PoolFun':
        return MLP1PoolFun(channels, classes, layers = layers, hidden_size = hidden_size, Pool = getattr(torch_geometric.nn.pool, config.get('Pool', 'global_max_pool')), p = config.get('p', 2), norm_dim = config.get('dim', 1))
    elif types == 'MLP1PoolNorm':
        return MLP1PoolNorm(channels, classes, layers = layers, hidden_size = hidden_size, Pool = getattr(torch_geometric.nn.pool, config.get('Pool', 'global_max_pool')), Norm = Norm)
    elif types == 'GCN3Pool':
        return GCN3Pool(channels, classes, layers = layers, hidden_size = hidden_size)
    else:
        raise NotImplementedError