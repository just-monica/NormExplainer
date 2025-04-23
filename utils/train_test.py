import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from torchmetrics.functional import accuracy, precision, recall, f1_score

def train_graph(model, train_loader, optimizer, criterion, device):
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(x = (data.x), edge_index = (data.edge_index), batch = data.batch)
        '''
        if data.y.dim()>1:
            data.y = data.y.squeeze().long()
        '''
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

def test_graph(model, test_loader, criterion, device):
    '''
    test the model on the test set
    return the total accuracy and the total loss
    '''
    model.eval()
    correct = 0
    loss = 0.0
    tp = 0 #true positive
    pp = 0 #predicted positive
    fn = 0 #false negative
    for data in test_loader:
        data = data.to(device)
        out = model(x = data.x, edge_index = data.edge_index, batch = data.batch)
        '''
        if data.y.dim()>1:
            data.y = data.y.squeeze().long()
        '''
        loss += criterion(out, data.y).item() * data.num_graphs
        pred = out.argmax(dim=1).to(device)
        correct += int((pred == data.y).sum())
        tp += int(((pred == 1) & (data.y == 1)).sum())
        fn += int(((pred == 0) & (data.y == 1)).sum())
        pp += int((pred == 1).sum())

    precision = tp/pp if pp != 0 else 0.0
    recall = tp/(tp+fn) if (tp+fn) != 0 else 0.0
    f1 = 2 * precision * recall/(precision + recall) if (precision + recall) != 0 else 0.0
    num_graphs = len(test_loader.dataset)
    return correct/num_graphs, precision, -loss/num_graphs, recall, f1 #return negative loss




def train_node(model, data, train_idx, optimizer, criterion):#data should already be on device
    model.train()
    optimizer.zero_grad()
    out = model(x = data.x, edge_index = data.edge_index)
    loss = criterion(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    return model
    
def test_node(model, data, test_idx, criterion):
    model.eval()
    out = model(x = data.x, edge_index = data.edge_index)
    loss = criterion(out[test_idx], data.y[test_idx])
    pred = out.argmax(dim=-1)
    correct = (pred[test_idx] == data.y[test_idx]).int().sum()
    return correct/len(test_idx), loss
'''
def test_graph2(model, test_loader, criterion, device, ):
    model.eval()
    correct = 0
    loss = 0.0
    for data in test_loader:
        data = data.to(device)
        edge_weight = torch.ones(data.edge_index.size(1)).float().to(device)
        out = model(x = data.x.to(device), edge_index = data.edge_index.to(device), batch = data.batch, edge_weight = edge_weight)
        if data.y.dim()>1:
            data.y = data.y.squeeze().long()
        loss += criterion(out, data.y).item() * data.num_graphs
        pred = out.argmax(dim=1).to(device)
        correct += int((pred == data.y).sum())
    return correct, loss
'''