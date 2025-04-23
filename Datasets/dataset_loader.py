import pickle as pkl
import numpy as np
import os
from numpy.random.mtrand import RandomState
import torch
import sys
from Datasets.DatasetsFromDIG.SynDatasets import SynDataset
from Datasets.DatasetsFromDIG.Mutagenicity import Mutagenicity, MutagenicityNoFeature
from Datasets.DatasetsFromDIG.BA2Motifs import BA2Motifs
from Datasets.DatasetsFromPyG.Benzene import Benzene
from Datasets.DatasetsFromPyG.AlkaneCarbonyl import AlkaneCarbonyl
from Datasets.DatasetsFromPyG.FluorideCarbonyl import FluorideCarbonyl
from Datasets.DatasetsFromPyG.mnist import MNIST75sp, MNIST75spAll
from Datasets.DatasetsFromPyG.NeoBA2Motifs import NeoBA2Motifs
from Datasets.DatasetsFromDIG.NeoMutagenicity import NeoMutagenicity
def dataset_selector(dataset_name = 'ba2motifs', folder = 'dig'):
    '''
    load dataset from the folder DatasetsFromDIG, DatasetsFromNumpy,  LargerBA2Motif, LargerBA3Motif,
    all of the datasets are saved as .pt files
    param: dataset_name: the name of the dataset, choose from 'ba2motifs', 'bashapes', 'bacommunity', 'treecycle', 'treegrid', to load larger datasets such as ba2_80, choose from 'ba2' and 'ba3'
    param: num_nodes: the number of nodes in the BA graph, only used for the larger ba2motifs and ba3motifs dataset
    param: folder: the folder where the dataset is saved, choose from 'dig', 'numpy', 'larger'
    '''
    folders = {'dig': 'DatasetsFromDIG', 'numpy': 'DatasetsFromNumpy', 'larger': 'Larger'}
    names = {'ba2motifs': 'BA2Motifs', 'bashapes': 'BAShapes', 'bacommunity' : 'BACommunity', 
             'treecycle' : 'TreeCycle', 'treegrid' : 'TreeGrid', 'ba2': 'BA2Motifs', 'ba3': 'BA3Motifs', 'bax': 'BAXMotifs'}
    dig_names = {'ba2motifs': 'ba_2motifs', 'bashapes': 'ba_shapes', 'bacommunity' : 'ba_community', 
             'treecycle' : 'tree_cycle', 'treegrid' : 'tree_grid', 'mutagenicity': 'mutagenicity'}
    #load the datasets in folders DatasetsFromDIG, DatasetsFromNumpy and LargerBA2Motif
    if folder == 'dig':
        current_path = os.path.dirname(os.path.realpath(__file__))
        #path = os.path.join(current_path, f'DatasetsFromDIG/{names[dataset_name]}.pt')
        if dataset_name.lower() == 'mutagenicity':
            dataset = Mutagenicity(root = os.path.join(current_path, 'DatasetsFromDIG'), name = dig_names[dataset_name.lower()])
        else:
            dataset = SynDataset(root = os.path.join(current_path, 'DatasetsFromDIG'), name = dig_names[dataset_name])


    else:
        current_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(current_path, f'{folders[folder]}/{names[dataset_name]}.pt')

    return dataset

def new_dataset_selector(dataset_name = 'ba2motifs', resample = None):
    current_path = os.path.dirname(os.path.realpath(__file__))
    if dataset_name == 'ba2':
        dataset = BA2Motifs(root = os.path.join(current_path, 'DatasetsFromDIG'), name = 'BA2Motifs')
    elif dataset_name == 'neoba2':
        dataset = NeoBA2Motifs(root = os.path.join(current_path, 'DatasetsFromPyG'), name = 'NeoBA2Motifs')
    elif dataset_name == 'mnist':
        dataset = MNIST75spAll(root = os.path.join(current_path, 'DatasetsFromPyG', 'MNIST75SP'))
    elif dataset_name == 'benzene':
        dataset = Benzene(root = os.path.join(current_path, 'DatasetsFromPyG'), name = 'benzene')
    elif dataset_name == 'ac':
        dataset = AlkaneCarbonyl(root = os.path.join(current_path, 'DatasetsFromPyG'), name = 'AlkaneCarbonyl')
        if resample is not None:
            positive_idx = torch.where(dataset.y == 1)[0]
            negative_idx = torch.where(dataset.y == 0)[0]
            num_positive = len(positive_idx)
            num_negative = len(negative_idx)
            select_negative_idx = torch.randperm(num_negative)[:int(num_positive * resample)]
            select_negative_idx = negative_idx[select_negative_idx]
            dataset = dataset[torch.cat([positive_idx, select_negative_idx])]
    elif dataset_name == 'fc':
        dataset = FluorideCarbonyl(root = os.path.join(current_path, 'DatasetsFromPyG'), name = 'FluorideCarbonyl')
        if resample is not None:
            positive_idx = torch.where(dataset.y == 1)[0]
            negative_idx = torch.where(dataset.y == 0)[0]
            num_positive = len(positive_idx)
            num_negative = len(negative_idx)
            select_negative_idx = torch.randperm(num_negative)[:int(num_positive * resample)]
            select_negative_idx = negative_idx[select_negative_idx]
            dataset = dataset[torch.cat([positive_idx, select_negative_idx])]
    elif dataset_name == 'mutagenicity':
        dataset = Mutagenicity(root = os.path.join(current_path, 'DatasetsFromDIG'), name = 'mutagenicity')
    elif dataset_name == 'mutagenicity_no_feature':
        dataset = MutagenicityNoFeature(root = os.path.join(current_path, 'DatasetsFromDIG'), name = 'mutagenicity_no_feat')
    elif dataset_name == 'neomutagenicity':
        dataset = NeoMutagenicity(root = os.path.join(current_path, 'DatasetsFromDIG'), name = 'neomutagenicity')
    else:
        raise ValueError(f'dataset {dataset_name} not found')
    return dataset
'''
def explain_dataset(dataset_name = 'mutagenicity'):
    if dataset_name == 'ba2':
        dataset = BA2Motifs(root = os.path.join(current_path, 'DatasetsFromDIG'), name = 'BA2Motifs')
        explain_dataset = dataset
    elif dataset_name == 'mnist':
        explain_dataset = dataset
    elif dataset_name == 'benzene':
        target_label = 1
        dataset = Benzene(root = os.path.join(current_path, 'DatasetsFromPyG'), name = 'benzene')
        target_idx = torch.where(dataset.y == target_label)[0]
        explain_dataset = dataset[target_idx]
    elif dataset_name == 'ac':
        target_label = 1
        dataset = AlkaneCarbonyl(root = os.path.join(current_path, 'DatasetsFromPyG'), name = 'AlkaneCarbonyl')
        target_idx = torch.where(dataset.y == target_label)[0]
        explain_dataset = dataset[target_idx]
    elif dataset_name == 'fc':
        target_label = 1
        dataset = FluorideCarbonyl(root = os.path.join(current_path, 'DatasetsFromPyG'), name = 'FluorideCarbonyl')
        target_idx = torch.where(dataset.y == target_label)[0]
        explain_dataset = dataset[target_idx]
    elif dataset_name == 'mutagenicity':
        target_label = 0
        dataset = Mutagenicity(root = os.path.join(current_path, 'DatasetsFromDIG'), name = 'mutagenicity')
        target_idx = torch.where(dataset.y == target_label)[0]
        explain_dataset = dataset[target_idx]
    elif dataset_name == 'mutagenicity_no_feature':
        target_label = 0
        dataset = MutagenicityNoFeature(root = os.path.join(current_path, 'DatasetsFromDIG'), name = 'mutagenicity_no_feat')
        target_idx = torch.where(dataset.y == target_label)[0]
        explain_dataset = dataset[target_idx]
    else:
        raise ValueError(f'dataset {dataset_name} not found')
    return explain_dataset
'''
def explain_dataset(dataset_name = 'mutagenicity', split_ratio = 0.8):
    dataset = new_dataset_selector(dataset_name)
    if split_ratio is not None:
        if dataset_name in ['ba2', 'neoba2']:
            length = len(dataset)
            idx = torch.randperm(length)
            train_idx = idx[:int(length * split_ratio)]
            test_idx = idx[int(length * split_ratio):]
            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx]
        elif dataset_name in ['benzene', 'ac', 'fc']:
            target_label = 1
            target_idx = torch.where(dataset.y == target_label)[0]
            dataset = dataset[target_idx]
            length = len(dataset)
            idx = torch.randperm(length)
            train_idx = idx[:int(length * split_ratio)]
            test_idx = idx[int(length * split_ratio):]
            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx]
        elif dataset_name in ['mutagenicity', 'mutagenicity_no_feature', 'neomutagenicity']:
            target_label = 0
            target_idx = torch.where(dataset.y == target_label)[0]
            dataset = dataset[target_idx]
            length = len(dataset)
            idx = torch.randperm(length)
            train_idx = idx[:int(length * split_ratio)]
            test_idx = idx[int(length * split_ratio):]
            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx]
        elif dataset_name == 'mnist':
            length = len(dataset)
            rand_idx = torch.randperm(length)[:10000]
            train_idx = rand_idx[:int(10000 * split_ratio)]
            test_idx = rand_idx[int(10000 * split_ratio):]
            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx]
        else:
            raise ValueError(f'dataset {dataset_name} not found')
        return train_dataset, test_dataset
    else:
        if dataset_name in ['ba2', 'ba3, neoba2']:
            pass
        elif dataset_name in ['benzene', 'ac', 'fc']:
            target_label = 1
            target_idx = torch.where(dataset.y == target_label)[0]
            dataset = dataset[target_idx]
        elif dataset_name in ['mutagenicity', 'mutagenicity_no_feature', 'neomutagenicity']:
            target_label = 0
            target_idx = torch.where(dataset.y == target_label)[0]
            dataset = dataset[target_idx]
        elif dataset_name in ['mnist']: # only use 10000 samples for explanation task
            length = len(dataset)
            rand_idx = torch.randperm(length)[:10000]
            dataset = dataset[rand_idx]
        else:
            raise ValueError(f'dataset {dataset_name} not found')
        return dataset