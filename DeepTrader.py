import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class DeepTrader_Model():
    def __init__(self):
        super(DeepTrader_Model, self).__init__()

    def forward(self, inputs):
        return inputs

def loadDeepTrader_Model(fn = "deeptrader_model.pt"):
    model = DeepTrader_Model()
    model.load_state_dict(torch.load(fn))
    return model

def train(num_epochs, data_loader, device=torch.device('cpu')):
    model = DeepTrader_Model()
    optimizer = optim.Adam()
    for e in num_epochs:
        _train_epoch(model, data_loader, optimizer, device)
    torch.save(model.state_dict(), "deeptrader_model.pt")

def _train_epoch(model, data_loader, optimizer, device=torch.device('cpu')):
    loss_func = nn.CrossEntropyLoss()
    losses = []
    for batch, target in data_loader:
        optimizer.zero_grad()
        output = model(batch.float())
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, np.mean(losses)