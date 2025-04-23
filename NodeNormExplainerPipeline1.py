import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import torch_geometric.transforms as T
from torch_geometric.explain import Explainer
from Explainers.NormExplainer import NodeNormExplainer
from Datasets.dataset_loader import explain_dataset
from GNNs.model_selector import new_model_selector
from utils.explain_task import train_test_explainer
from torch_geometric.loader import DataLoader
from torchmetrics.functional import auroc, average_precision
from utils.metrics import binary_mask_evaluation as bme
import argparse
from datetime import datetime
import json
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=88)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--model_id', type=str, default='0914004626')
    parser.add_argument('--dataset', type=str, default='mutagenicity')
    parser.add_argument('--add_id', type=str, default='1')
    parser.add_argument('--sparsity_test', type=bool, default=False)
    parser.add_argument('--save_explainer', type = int, default=0)
    return parser.parse_args()


def main():
    bold_start = '\033[1m'
    bold_end = '\033[0m'
    time = datetime.now().strftime('%m%d%H%M')
    args = arg_parser()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed ** 2)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else 'cpu')
    model_config = yaml.safe_load(open('Configs/ModelConfigs/'+args.dataset+'_'+args.model_id+'.yaml', 'r'))
    task_config = yaml.safe_load(open('Configs/TaskConfigs/NodeNormExplainer/'+args.dataset+'_nodeexplainer'+args.add_id+'.yaml', 'r'))
    print(f'the task config is {task_config}')
    model = new_model_selector(task_config['dataset_name'], model_config['layers'], model_config['hidden_size'], model_config['types'], config=model_config)
    model.load_state_dict(torch.load(f"SavedModels/{model_config['saved_path']}"))
    model = model.to(device)
    if task_config['filter_dataset']:
        dataset = explain_dataset(args.dataset, None)
        correct_idx = json.load(open(f'./SavedData/explain_dataset_filter_{args.dataset}_{args.model_id}.json', 'r'))
        print(f'the length of the dataset is {len(dataset)}')
        dataset = dataset[correct_idx]
        print(f'the length of the dataset after filtering is {len(dataset)}')
        idx = torch.randperm(len(dataset))
        train_idx = idx[:int(len(dataset) * task_config['split_ratio'])]
        test_idx = idx[int(len(dataset) * task_config['split_ratio']):]
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
    else:
        train_dataset, test_dataset = explain_dataset(args.dataset, task_config['split_ratio'])
    print(f'the length of train dataset is {len(train_dataset)}')
    print(f'the length of test dataset is {len(test_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=task_config.get('batch_size', len(train_dataset)), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size= len(test_dataset), shuffle=False)



    algo = NodeNormExplainer(epochs= task_config['epoch'], lr = task_config['lr'], config=task_config, model=model, device=device)
    explainer = Explainer(
        model=model,
        algorithm=algo,
        explanation_type=task_config.get('explanation_type'),#model or phenomenon
        node_mask_type= task_config['node_mask_type'],
        edge_mask_type=task_config['edge_mask_type'],
        model_config=dict(
            mode=task_config.get('mode', 'multiclass_classification'),
            task_level=task_config.get('task_level', 'graph'),
            return_type=task_config.get('return_type', 'raw'),
        )
    )
    best_edge_auroc = 0
    best_node_auroc = 0
    #train the algo
    for i in range(task_config['epoch']):
        edge_mask_result, node_mask_result, edge_gt_result, node_gt_result = \
                    train_test_explainer(algo, train_loader, device, i, train_test = 'train')

        edge_auroc = auroc(edge_mask_result.cpu(), edge_gt_result.cpu(), task='binary')
        edge_ap = average_precision(edge_mask_result.cpu(), edge_gt_result.cpu(), task='binary')
        node_auroc = auroc(node_mask_result.cpu(), node_gt_result.cpu(), task='binary')
        node_ap = average_precision(node_mask_result.cpu(), node_gt_result.cpu(), task='binary')
        print(f'{bold_start}in the train set the edge auroc is {edge_auroc}, the edge ap is {edge_ap}, the node auroc is {node_auroc}, the node ap is {node_ap}{bold_end}')
        edge_mask_result, node_mask_result, edge_gt_result, node_gt_result = \
                    train_test_explainer(algo, test_loader, device, i, train_test = 'test')
        edge_auroc = auroc(edge_mask_result.cpu(), edge_gt_result.cpu(), task='binary')
        node_auroc = auroc(node_mask_result.cpu(), node_gt_result.cpu(), task='binary')
        edge_ap = average_precision(edge_mask_result.cpu(), edge_gt_result.cpu(), task='binary')
        node_ap = average_precision(node_mask_result.cpu(), node_gt_result.cpu(), task='binary')
        print(f'{bold_start}in the test set the edge auroc is {edge_auroc}, the edge ap is {edge_ap}, the node auroc is {node_auroc}, the node ap is {node_ap}{bold_end}')



    print(f"Save explainer flag is set to: {args.save_explainer}")
    if args.save_explainer == 1:
        torch.save(algo.node_mlp.state_dict(), f'SavedExplainer/NormExplainer_{args.dataset}_{args.model_id}.pth')
        print('explainer saved')
    if args.sparsity_test:
        sparsity_list = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        print(f'{bold_start}start sparsity test{bold_end}')
        bme(edge_mask_result, node_mask_result, edge_gt_result, node_gt_result, sparsity_list)





if __name__ == '__main__':
    main()

'''
CUDA_VISIBLE_DEVICES=0 python NodeNormExplainerPipeline1.py --seed 6 --dataset mutagenicity --model_id 0107150753
'''
#this pipeline should be used for mutagenicity dataset