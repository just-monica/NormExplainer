import argparse
import os
import os.path as osp
import time
import sys
sys.path.append("..")
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Softmax, Sequential

from torch_geometric.nn import BatchNorm,  InstanceNorm, LayerNorm
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

EPS = 1
class GCN3PoolNorm(torch.nn.Module):
    '''
    use the normalization layer in the torch_geometric.nn such as BatchNorm, InstanceNorm, LayerNorm
    '''
    def __init__(self, num_features, num_classes, layers = 3, hidden_size = 20, Norm = BatchNorm, dropout = 0.0):
        super(GCN3PoolNorm, self).__init__()
        self.embedding_size = hidden_size
        self.layers = layers
        self.dropout = dropout
        self.convs = ModuleList()
        self.relus = ModuleList()
        self.norms = ModuleList()
        self.convs.append(GCNConv(in_channels=num_features, out_channels=hidden_size))
        self.convs.extend([GCNConv(in_channels=hidden_size, out_channels=hidden_size)] * (self.layers - 1))
        self.relus.extend([ReLU()] * self.layers)
        self.norms.extend([Norm(hidden_size)] * self.layers)
        self.lin = Sequential(Linear(3*hidden_size , hidden_size), ReLU(), Linear(hidden_size, num_classes))
        forward_signature = inspect.signature(self.norms[0].forward)#check the first norm's forward()
        self.has_batch = 'batch' in forward_signature.parameters
        print(f'using {self.norms[0].__class__.__name__} as the normalization layer')
    def forward(self, x, edge_index, batch=None, edge_attr=None):
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.get_graph_embedding(x, edge_index, batch, edge_attr)

        out = self.lin(embed)
        return out

    def get_node_embedding(self, x, edge_index, batch=None, edge_attr=None):
        out = x
        for conv, norm, relu in zip(self.convs, self.norms, self.relus):
            out = conv(out, edge_index)
            if self.has_batch:
                out = norm(out, batch)
            else:
                out = norm(out)
            out = relu(out)
            #out = F.dropout(out, p=self.dropout, training=self.training)
        return out
    
    def get_graph_embedding(self, x, edge_index, batch, edge_attr=None):#get the graph embedding
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.get_node_embedding(x, edge_index, batch, edge_attr)
        out1 = global_max_pool(embed, batch)
        out2 = global_mean_pool(embed, batch)
        out3 = global_add_pool(embed, batch)
        out = torch.cat([out1, out2, out3], dim = -1)
        return out
    
    def set_dropout(self, dropout):
        self.dropout = dropout

class GCN3PoolFun(torch.nn.Module):
    '''
    use the normalization function in the torch.nn.functional
    '''
    def __init__(self, num_features, num_classes, layers = 3, hidden_size = 20, dropout = 0.0, norm_dim = 1, p=2):
        super(GCN3PoolFun, self).__init__()
        self.embedding_size = hidden_size
        self.layers = layers
        self.dropout = dropout
        self.norm_dim = norm_dim
        self.p = p
        self.convs = ModuleList()
        self.relus = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(GCNConv(in_channels=num_features, out_channels=hidden_size))
        self.convs.extend([GCNConv(in_channels=hidden_size, out_channels=hidden_size)] * (self.layers - 1))
        self.relus.extend([ReLU()] * self.layers)
        self.lin = Sequential(Linear(3*hidden_size , hidden_size), ReLU(), Linear(hidden_size, num_classes))
        print(f'using norm_dim = {self.norm_dim} and p = {self.p} for normalization')
    def forward(self, x, edge_index, batch=None, edge_attr=None):
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.get_graph_embedding(x, edge_index, batch, edge_attr)
        out = self.lin(embed)
        return out

    def get_node_embedding(self, x, edge_index, batch=None, edge_attr=None):
        out = x
        for conv, relu in zip(self.convs, self.relus):
            out = conv(out, edge_index)
            out = F.normalize(out, p=self.p, dim=self.norm_dim)
            out = relu(out)
            #out = F.dropout(out, p=self.dropout, training=self.training)
        return out
    
    def get_graph_embedding(self, x, edge_index, batch, edge_attr=None):#get the graph embedding
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.get_node_embedding(x, edge_index, batch, edge_attr)
        out1 = global_max_pool(embed, batch)
        out2 = global_mean_pool(embed, batch)
        out3 = global_add_pool(embed, batch)
        out = torch.cat([out1, out2, out3], dim = -1)
        return out
    
    def set_dropout(self, dropout):
        self.dropout = dropout

class GCN1PoolNorm(torch.nn.Module):
    '''
    use the normalization layer in the torch_geometric.nn such as BatchNorm, InstanceNorm, LayerNorm
    '''
    def __init__(self, num_features, num_classes, layers = 3, hidden_size = 20, Norm = BatchNorm, dropout = 0.0, Pool = global_max_pool):
        super(GCN1PoolNorm, self).__init__()
        self.embedding_size = hidden_size
        self.layers = layers
        self.dropout = dropout
        self.Pool = Pool
        self.convs = ModuleList()
        self.relus = ModuleList()
        self.norms = ModuleList()
        self.convs.append(GCNConv(in_channels=num_features, out_channels=hidden_size))
        self.convs.append(GCNConv(in_channels=hidden_size, out_channels=hidden_size))
        self.convs.append(GCNConv(in_channels=hidden_size, out_channels=hidden_size))
        self.relus.extend([ReLU()] * self.layers)
        self.norms.extend([Norm(hidden_size)] * self.layers)
        self.lin = Sequential(Linear(hidden_size , hidden_size), ReLU(), Linear(hidden_size, num_classes))
        forward_signature = inspect.signature(self.norms[0].forward)#check the first norm's forward()
        self.has_batch = 'batch' in forward_signature.parameters
        print(f'using {self.norms[0].__class__.__name__} as the normalization layer')
        print(f'using {self.Pool.__name__} as the pooling layer')
    def forward(self, x, edge_index, batch=None, edge_attr=None):
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.get_graph_embedding(x, edge_index, batch, edge_attr)

        out = self.lin(embed)
        return out

    def get_node_embedding(self, x, edge_index, batch=None, edge_attr=None):
        out = x
        for conv, norm, relu in zip(self.convs, self.norms, self.relus):
            out = conv(out, edge_index)
            if self.has_batch:
                out = norm(out, batch)
            else:
                out = norm(out)
            out = relu(out)
            #out = F.dropout(out, p=self.dropout, training=self.training)
        return out
    
    def get_graph_embedding(self, x, edge_index, batch, edge_attr=None):#get the graph embedding
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.get_node_embedding(x, edge_index, batch, edge_attr)
        out = self.Pool(embed, batch)
        return out
    
    def set_dropout(self, dropout):
        self.dropout = dropout

class GCN1Pool(torch.nn.Module):
    '''
    use the normalization function in the torch.nn.functional
    '''
    def __init__(self, num_features, num_classes, layers = 3, hidden_size = 20, dropout = 0.0, Norm = LayerNorm, Pool = global_max_pool):
        super(GCN1Pool, self).__init__()
        self.embedding_size = hidden_size
        self.layers = layers
        self.dropout = dropout
        self.Pool = Pool
        self.convs = ModuleList()
        self.relus = ModuleList()
        self.norms = ModuleList()
        self.convs.append(GCNConv(in_channels=num_features, out_channels=hidden_size))
        self.convs.append(GCNConv(in_channels=hidden_size, out_channels=hidden_size))
        self.convs.append(GCNConv(in_channels=hidden_size, out_channels=hidden_size))
        self.relus.append(ReLU())
        self.relus.append(ReLU())
        self.relus.append(ReLU())
        self.norms.append(LayerNorm(hidden_size))
        self.norms.append(LayerNorm(hidden_size))
        self.norms.append(LayerNorm(hidden_size))
        self.lin = Sequential(Linear(hidden_size , hidden_size), ReLU(), Linear(hidden_size, num_classes))
        forward_signature = inspect.signature(self.norms[0].forward)#check the first norm's forward()
        self.has_batch = 'batch' in forward_signature.parameters
        print(f'using {self.norms[0].__class__.__name__} as the normalization layer')
        print(f'using {self.Pool.__name__} as the pooling layer')
    def forward(self, x, edge_index, batch=None, edge_attr=None):
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.get_graph_embedding(x, edge_index, batch, edge_attr)
        out = self.lin(embed)
        return out

    def get_node_embedding(self, x, edge_index, batch=None, edge_attr=None):
        out = x
        for conv, norm, relu in zip(self.convs, self.norms, self.relus):
            out = conv(out, edge_index)
            if self.has_batch:
                out = norm(out, batch)
            else:
                out = norm(out)
            out = relu(out)
            #out = F.dropout(out, p=self.dropout, training=self.training)
        return out
    
    def get_graph_embedding(self, x, edge_index, batch, edge_attr=None):#get the graph embedding
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.get_node_embedding(x, edge_index, batch, edge_attr)
        out = self.Pool(embed, batch)
        return out
    
    def set_dropout(self, dropout):
        self.dropout = dropout


class GCN3Pool(torch.nn.Module):
    '''
    no normalization version
    '''
    def __init__(self, num_features, num_classes, layers = 3, hidden_size = 20,  dropout = 0.0):
        super(GCN3Pool, self).__init__()
        self.embedding_size = hidden_size
        self.layers = layers
        self.dropout = dropout
        self.convs = ModuleList()
        self.relus = ModuleList()
        self.convs.append(GCNConv(in_channels=num_features, out_channels=hidden_size))
        self.convs.extend([GCNConv(in_channels=hidden_size, out_channels=hidden_size)] * (self.layers - 1))
        self.relus.extend([ReLU()] * self.layers)
        self.lin = Sequential(Linear(3*hidden_size , num_classes))

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.get_graph_embedding(x, edge_index, batch, edge_attr)

        out = self.lin(embed)
        return out

    def get_node_embedding(self, x, edge_index, batch=None, edge_attr=None):
        out = x
        for conv, relu in zip(self.convs, self.relus):
            out = conv(out, edge_index)
            out = relu(out)
            #out = F.dropout(out, p=self.dropout, training=self.training)
        return out
    def get_graph_embedding(self, x, edge_index, batch, edge_attr=None):#get the graph embedding
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.get_node_embedding(x, edge_index, batch, edge_attr)
        out1 = global_max_pool(embed, batch)
        out2 = global_mean_pool(embed, batch)
        out3 = global_add_pool(embed, batch)
        out = torch.cat([out1, out2, out3], dim = -1)
        return out