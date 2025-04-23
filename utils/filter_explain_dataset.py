import torch
import json
import sys
import os
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import torch_geometric.transforms as T
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Datasets.dataset_loader import explain_dataset
from GNNs.model_selector import new_model_selector
from torch_geometric.loader import DataLoader

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=88)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--model_id', type=str, default='0107150753')
    parser.add_argument('--dataset', type=str, default='mutagenicity')
    return parser.parse_args()

def filter_explain_dataset():
    args = arg_parser()
    dataset = explain_dataset(args.dataset, None)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else 'cpu')
    model_config = yaml.safe_load(open('Configs/ModelConfigs/'+args.dataset+'_'+args.model_id+'.yaml', 'r'))
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    model = new_model_selector(args.dataset, model_config['layers'], model_config['hidden_size'], model_config['types'], config=model_config)
    model.load_state_dict(torch.load(os.path.join('SavedModels', model_config['saved_path'])))
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch, data.edge_attr)
            pred = output.argmax(dim=-1)
            correct_positions = (pred == data.y).nonzero().view(-1)
            print(f'the number of correct positions is {correct_positions.size(0)}')
            print(f'the number of total positions is {data.y.size(0)}')
            print(f'the ratio of the correct positions is {correct_positions.size(0)/data.y.size(0)}')
    
    with open(f'./SavedData/explain_dataset_filter_{args.dataset}_{args.model_id}.json', 'w', encoding='utf-8') as f:
        json.dump(correct_positions.tolist(), f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    filter_explain_dataset()