import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib as plt

# class DeepTrader_Model():
#     def __init__(self):
#         super(DeepTrader_Model, self).__init__()

#     def forward(self, inputs):
#         return inputs

class DeepTrader_Model(nn.Module):
    def __init__(self):
        super(DeepTrader_Model, self).__init__()
        
        # LSTM layer 
        self.lstm = nn.LSTM(13, 10)
        
        # # Dropout layer
        # self.drop = nn.Dropout(p=0.2)
        
        # First fully connected layer
        self.fc1 = nn.Linear(10, 5)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(5, 3)
        
        # Third fully connected layer - output
        self.fc3 = nn.Linear(3, 1)
        
    # x represents our data
    def forward(self, x):
        
        # Pass data through lstm layer
        x = self.lstm(x)
        # x = self.drop(x)
        
        # Use the rectified-linear activation function over x
        x = F.relu(x)

        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        # output = F.relu(x)
        

        # # Apply softmax to x
        output = F.softmax(x, dim=1)
        return output

model = DeepTrader_Model()

def saveDeepTrader_Model(model, fn = 'deeptrader_model.pt'):
    torch.save(model.state_dict(), fn)

def loadDeepTrader_Model(fn = 'deeptrader_model.pt'):
    model = DeepTrader_Model()
    model.load_state_dict(torch.load(fn))
    return model

def train(num_epochs, data_loader, device=torch.device('cpu')):
    model = DeepTrader_Model()
    optimizer = optim.Adam(model.parameters(), lr=1.5e-5)
    losses = np.array([])
    for e in range(num_epochs):
        model, elosses = _train_epoch(model, data_loader, optimizer, device)
        losses.append(np.mean(elosses))
    torch.save(model.state_dict(), "deeptrader_model.pt")
    plt.plot(losses)
    plt.show()

def _train_epoch(model, data_loader, optimizer, device=torch.device('cpu')):
    loss_func = nn.MSELoss()
    losses = []
    for batch, target in data_loader:
        optimizer.zero_grad()
        output = model(batch.float())
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, np.mean(losses)
    
