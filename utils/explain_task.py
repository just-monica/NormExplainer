import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torchmetrics.functional import auroc
import argparse

def train_test_explainer(algothrim, train_loader, device, i, train_test = 'train'):
    edge_mask_result = []
    node_mask_result = []
    edge_gt_result = []
    node_gt_result = []
    for data in train_loader:
        data = data.to(device)
        edge_gt_result.append(data.edge_mask)
        node_gt_result.append(data.node_label)
        if train_test == 'train':
            _, loss_dict = algothrim.train(i, data.x, data.edge_index, edge_attr = data.edge_attr, batch = data.batch, index=None, target=data.y)
            print(f'in epoch {i}, the loss is {loss_dict}')
        
        explanation = algothrim(data.x, data.edge_index, edge_attr = data.edge_attr)
        edge_mask_result.append(explanation.edge_mask)
        if explanation.node_mask is not None:
            node_mask_result.append(explanation.node_mask.view(-1))
    edge_mask_result = torch.cat(edge_mask_result, dim=0)
    node_mask_result = torch.cat(node_mask_result, dim=0) if len(node_mask_result) > 0 else None
    edge_gt_result = torch.cat(edge_gt_result, dim=0).long()
    node_gt_result = torch.cat(node_gt_result, dim=0).long()

    return edge_mask_result, node_mask_result, edge_gt_result, node_gt_result
'''
def test_explainer(algothrim, test_loader, device):
    edge_mask_result = []
    node_mask_result = []
    edge_gt_result = []
    node_gt_result = []

    for data in test_loader:
        data = data.to(device)
        edge_gt_result.append(data.edge_mask)
        node_gt_result.append(data.node_label)
        explanation = algothrim(data.x, data.edge_index, edge_attr = data.edge_attr)
        edge_mask_result.append(explanation.edge_mask)
        node_mask_result.append(explanation.node_mask.view(-1))
    edge_mask_result = torch.cat(edge_mask_result, dim=0)
    node_mask_result = torch.cat(node_mask_result, dim=0)
    edge_gt_result = torch.cat(edge_gt_result, dim=0).long()
    node_gt_result = torch.cat(node_gt_result, dim=0).long()

    return edge_mask_result, node_mask_result, edge_gt_result, node_gt_result
'''