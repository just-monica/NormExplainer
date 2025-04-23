import torch

def split_dataset(length, ratio = 0.8):
    shuffled_index = torch.randperm(length).tolist()
    train_end_index = int(length*ratio)
    train_index = shuffled_index[:train_end_index]
    test_index = shuffled_index[train_end_index:]
    return train_index, test_index