import torch

def top_k_subgraph(edge_mask, k):
    #assume edge_mask is a tensor of shape E
    flat_mask = edge_mask.view(-1)
    _, indices = torch.topk(flat_mask, k)
    mask = torch.zeros_like(flat_mask).float()
    mask[indices] = 1.0
    return mask.view(edge_mask.shape)

def top_k_node(node_mask, k, rule = 'sum'):#rule = 'sum' or 'max'
    #assume node_mask is a tensor of shape N*1
    flat_mask = node_mask.view(-1)
    _, indices = torch.topk(flat_mask, k)
    mask = torch.zeros_like(flat_mask).float()
    mask[indices] = 1.0
    return mask

def top_subgraph(mask, sparsity):
    #assume mask is a tensor of shape E or N
    k = int(mask.size(0) * (1 - sparsity))
    return top_k_subgraph(mask, k) 