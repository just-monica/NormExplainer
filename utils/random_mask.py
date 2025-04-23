import torch
from typing import Union
def random_edge_mask(ground_truth, edges_list):
    '''
        generate a random mask for the edges in the graph, the mask is a tensor with the same size as the edge_index
        and the value is 0 or 1, the number of 1 is the same as the number of edges in the graph
    '''
    mask = torch.zeros(ground_truth.size(0))
    
    masks = mask.split(edges_list)
    gts = ground_truth.split(edges_list)
    for gt, mask in zip(gts, masks):
        num_ones = gt.sum().int().item()
        ones_idx = torch.randperm(mask.size(0))[:num_ones]
        mask[ones_idx] = 1.0

    modified_mask = torch.cat(masks)
    return modified_mask.float()

def random_node_mask(ground_truth, nodes_list):
    '''
        generate a random mask for the nodes in the graph, the mask is a tensor with size N*1
        and the value is 0 or 1, the number of 1 is the same as the number of nodes in the graph
        ground_truth is a tensor with size N
    '''
    mask = torch.zeros(ground_truth.size(0))
    masks = mask.split(nodes_list)
    gts = ground_truth.split(nodes_list)
    for gt, mask in zip(gts, masks):
        num_ones = gt.sum().int().item()
        ones_idx = torch.randperm(mask.size(0))[:num_ones]
        mask[ones_idx] = 1.0

    modified_mask = torch.cat(masks).unsqueeze(1)
    return modified_mask.float()

if __name__ == '__main__':
    ground_truth0 = torch.zeros(10)
    ground_truth1 = torch.ones(3)
    ground_truth2 = torch.zeros(20)
    ground_truth3 = torch.ones(8)
    ground_truth = torch.cat([ground_truth0, ground_truth1, ground_truth2, ground_truth3])
    edges_list = [23, 18]
    gt_remove_num = 1
    modified_gt = random_edge_mask(ground_truth, edges_list)
    print(modified_gt)

def modify_binary_tensor(mask: torch.Tensor, p1: float, p2: float) -> torch.Tensor:
    """
    修改二值tensor中的0和1的值
    
    参数:
    m: 输入tensor，只包含0和1
    p1: 需要从1变为0的比例 (0 <= p1 <= 1)
    p2: 需要从0变为1的比例 (0 <= p2 <= 1)
    
    返回:
    修改后的tensor
    
    异常:
    ValueError: 当输入参数不符合要求时
    """
    mask = mask.clone().int()
    result = mask
    # 参数检查
    if not isinstance(mask, torch.Tensor):
        raise ValueError("输入必须是torch.Tensor类型")
        
    if not torch.all(torch.logical_or(mask == 0, mask == 1)):
        raise ValueError("输入tensor必须只包含0和1")
        
    if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
        raise ValueError("p1和p2必须在0到1之间")

    
    # 计算0和1的数量
    num_ones = torch.sum(mask == 1).item()
    num_zeros = torch.sum(mask == 0).item()
    
    # 计算需要修改的数量
    num_ones_to_change = int(num_ones * p1)
    num_zeros_to_change = int(num_zeros * p2)
    
    # 获取所有1和0的位置索引
    ones_indices = (mask == 1).nonzero(as_tuple=False)
    zeros_indices = (mask == 0).nonzero(as_tuple=False)
    
    # 随机修改1为0
    if num_ones_to_change > 0 and len(ones_indices) > 0:
        perm = torch.randperm(len(ones_indices))[:num_ones_to_change]
        selected_ones = ones_indices[perm]
        result[selected_ones] = 0
    
    # 随机修改0为1
    if num_zeros_to_change > 0 and len(zeros_indices) > 0:
        perm = torch.randperm(len(zeros_indices))[:num_zeros_to_change]
        selected_zeros = zeros_indices[perm]
        result[selected_zeros] = 1
    
    return result.float()
