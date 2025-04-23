import os
import torch
import pickle
import numpy as np
import os.path as osp
import sys
import tqdm
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.data.dataset import files_exist


def read_ba2motif_data(folder: str, prefix, reduction = False):
    with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
        dense_edges, node_features, graph_labels = pickle.load(f)

    data_list = []
    for graph_idx in range(dense_edges.shape[0]):
        edge_index = dense_to_sparse(torch.from_numpy(dense_edges[graph_idx]))[0]
        x = torch.from_numpy(node_features[graph_idx]).float()
        if reduction:
            x = x*0.1
        data_list.append(Data(x=x,
                              edge_index=edge_index,
                              y=torch.from_numpy(np.where(graph_labels[graph_idx])[0]).to(torch.long),
                              edge_attr = torch.ones((edge_index.size(1), 5)).float(), #dummy edge attributes, why 5? just the same as other datasets
                              edge_mask = BA2Motifs.gen_motif_edge_mask(edge_index = edge_index),
                              node_label = BA2Motifs.gen_motif_node_label(x = x)))
    return data_list


class BA2Motifs(InMemoryDataset):
    r"""
    the ba2motifs dataset from DIG with additional node_label attribute as the node ground truth labels

    Args:
        root (:obj:`str`): Root data directory to save datasets
        name (:obj:`str`): The name of the dataset. Including :obj:`BA_shapes`, BA_grid,
        transform (:obj:`Callable`, :obj:`None`): A function/transform that takes in an
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (:obj:`Callable`, :obj:`None`):  A function/transform that takes in
            an :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """

    url = 'https://github.com/divelab/DIG_storage/raw/main/xgraph/datasets/{}'
    # Format: name: [display_name, url_name, filename]
    names = {
        'BA2Motifs': ['BA_2Motifs', 'BA_2Motifs.pkl', 'BA_2Motifs', 'BA2Motifs']
    }

    def __init__(self, root, name, transform=None, pre_transform=None,  reduction =  False):
        self.name = name #just use the name as it is, BA2Motifs
        self.reduction = reduction
        super(BA2Motifs, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.pkl'

    @property
    def processed_file_names(self):
        if self.reduction:
            return [f'{self.names[self.name][3]}_reduced_data.pt']
        else:
            return [f'{self.names[self.name][3]}_data.pt']

    def download(self):
        url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)

    def process(self):
        data_list = read_ba2motif_data(self.raw_dir, self.names[self.name][2], self.reduction)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)


        torch.save(self.collate(data_list), self.processed_paths[0])#TODO

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))

    @staticmethod
    def gen_motif_edge_mask(edge_index, node_idx=0, num_hops=3):
        return torch.logical_and(edge_index[0] >= 20, edge_index[1] >= 20).to(torch.long)

    @staticmethod
    def gen_motif_node_label(x):
        node_label = torch.zeros(x.size(0)).to(torch.long)
        node_label[20:] = 1
        return node_label

    def read_syn_data(self):
        with open(self.raw_paths[0], 'rb') as f:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pickle.load(f)

        x = torch.from_numpy(features).float()
        y = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
        y = torch.from_numpy(np.where(y)[1])
        edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = torch.from_numpy(train_mask)
        data.val_mask = torch.from_numpy(val_mask)
        data.test_mask = torch.from_numpy(test_mask)
        data.edge_label_matrix = torch.from_numpy(edge_label_matrix)
        return data


if __name__ == '__main__':
    # TODO: add the test code here
    dataset = BA2Motifs(root = './', name = 'BA2Motifs', reduction = False)
    print(len(dataset))
    print(dataset.x.size())
    print(dataset[0].x)
    print(dataset.y.size())
    print(dataset.y.dtype)
    print(dataset.edge_index.size())
    print(dataset.edge_attr.size())
    print(dataset.edge_mask.size())
    print(dataset.edge_mask.sum())
    print(dataset.node_label.size())
    print(dataset.node_label.sum())