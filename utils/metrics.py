import torch
from torchmetrics import Precision, AUROC
from torchmetrics.classification import BinaryPrecision
from torchmetrics.functional import auroc, precision, recall, f1_score, accuracy
from  utils.top_k_subgraph import top_subgraph, top_k_subgraph
from torchmetrics.functional import average_precision
def calculate_precision_auroc(preds, targets, task= 'binary'):
    # 计算精确度
    precision_metric =  BinaryPrecision()
    precision = precision_metric(preds, targets)


    # 计算AUROC
    auroc_metric = AUROC(num_classes=2, task=task)
    auroc = auroc_metric(preds, targets)
    return precision, auroc

def explain_metrics(preds, targets, task= 'binary'):
    '''
    preds: binary tensor (float), should be binaried by the sparsity before using this function
    targets: binary tensor (int)
    '''
    accuracy_value = accuracy(preds, targets, average='micro', task = task)

    precision_value = precision(preds, targets, average='micro', task = task)


    recall_value = recall(preds, targets, average='micro', task = task)

    f1_value = f1_score(preds, targets, average='micro', task = task)
    return accuracy_value, precision_value, recall_value, f1_value

def binary_mask_evaluation(edge_mask, node_mask, edge_gt, node_gt, sparsity_list = [0.6, 0.7, 0.8, 0.9]):
    '''
    edge_mask: tensor of shape E, float
    node_mask: tensor of shape N, float
    edge_gt: tensor of shape E, int
    node_gt: tensor of shape N, int
    '''
    bold_start = '\033[1m'
    bold_end = '\033[0m'
    edge_scores = torch.zeros(len(sparsity_list)+1, 4)
    node_scores = torch.zeros(len(sparsity_list)+1, 4)
    if edge_gt is not None and edge_mask is not None:
        edge_auroc = auroc(edge_mask.cpu(), edge_gt.cpu(), task='binary')
        edge_ap = average_precision(edge_mask.cpu(), edge_gt.cpu(), task='binary')
        print(f'{bold_start} the edge auroc is {edge_auroc}, the edge ap is {edge_ap} {bold_end}')
        for i, sparsity in enumerate(sparsity_list):
            binary_edge_mask = top_subgraph(edge_mask, sparsity)
            edge_acc, edge_precision, edge_recall, edge_f1 = explain_metrics(binary_edge_mask, edge_gt)
            edge_scores[i, :] = torch.tensor([edge_acc, edge_precision, edge_recall, edge_f1])
            print(f'{bold_start} for sparsity {sparsity}, the edge acc is {edge_acc}, the edge precision is {edge_precision}, the edge recall is {edge_recall}, the edge f1 is {edge_f1}{bold_end}')
        
        num_edge_gt = edge_gt.sum().item()
        binary_edge_mask = top_k_subgraph(edge_mask, num_edge_gt)
        edge_acc, edge_precision, edge_recall, edge_f1 = explain_metrics(binary_edge_mask, edge_gt)
        edge_scores[len(sparsity_list), :] = torch.tensor([edge_acc, edge_precision, edge_recall, edge_f1])
        print(f'{bold_start} for top {num_edge_gt}, the edge acc is {edge_acc}, the edge precision is {edge_precision}, the edge recall is {edge_recall}, the edge f1 is {edge_f1}{bold_end}')
    else:
        edge_auroc = torch.tensor([0.0])
    
    if node_gt is not None and node_mask is not None:
        node_auroc = auroc(node_mask.cpu(), node_gt.cpu(), task='binary')
        node_ap = average_precision(node_mask.cpu(), node_gt.cpu(), task='binary')
        print(f'{bold_start} the node auroc is {node_auroc}, the node ap is {node_ap} {bold_end}')
        for i, sparsity in enumerate(sparsity_list):
            binary_node_mask = top_subgraph(node_mask, sparsity)
            node_acc, node_precision, node_recall, node_f1 = explain_metrics(binary_node_mask, node_gt)
            node_scores[i, :] = torch.tensor([node_acc, node_precision, node_recall, node_f1])
            print(f'{bold_start} for sparsity {sparsity}, the node acc is {node_acc}, the node precision is {node_precision}, the node recall is {node_recall}, the node f1 is {node_f1}{bold_end}')
        
        num_node_gt = node_gt.sum().item()
        binary_node_mask = top_k_subgraph(node_mask, num_node_gt)
        node_acc, node_precision, node_recall, node_f1 = explain_metrics(binary_node_mask, node_gt)
        node_scores[len(sparsity_list), :] = torch.tensor([node_acc, node_precision, node_recall, node_f1])
        print(f'{bold_start} for top {num_node_gt}, the node acc is {node_acc}, the node precision is {node_precision}, the node recall is {node_recall}, the node f1 is {node_f1}{bold_end}')
    else:
        node_auroc = torch.tensor([0.0])
    return edge_auroc, node_auroc, edge_scores, node_scores

