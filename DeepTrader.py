import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import progressbar
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
    def forward(self, x, states):
        
        # Pass data through lstm layer
        # print(x)
        # print(x.shape)
        x, states = self.lstm(x, states)
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
        # output = F.softmax(x)
        return x, states

    def generate_initial_states(self, batch_size=None, device=torch.device('cpu')):
        return torch.zeros(1, batch_size, 10, device=device), torch.zeros(1, batch_size, 10, device=device)

    def detach_states(self, states):
        h, c = states
        return (h.detach(), c.detach())

# model = DeepTrader_Model()

def saveDeepTrader_Model(model, fn = 'deeptrader_model.pt'):
    torch.save(model.state_dict(), fn)

def loadDeepTrader_Model(fn = 'deeptrader_model.pt'):
    model = DeepTrader_Model()
    model.load_state_dict(torch.load(fn))
    return model

def train(num_epochs, data_loader, device=torch.device('cpu')):
    model = DeepTrader_Model()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    # print(list(model.parameters()))
    losses = []
    for e in range(num_epochs):
        print("running epoch "+str(e)+"...")
        model, elosses = _train_epoch(model, data_loader, optimizer, device)
        losses.append(np.mean(elosses))
    torch.save(model.state_dict(), "deeptrader_model.pt")
    plt.plot(losses)
    plt.show()

def _train_epoch(model, dataset, optimizer, device=torch.device('cpu')):
    loss_func = nn.MSELoss()
    losses = []
    counter = 0
    number_of_batches = 634
    states = model.generate_initial_states(dataset.batch_size, device)
    with progressbar.ProgressBar(max_value = number_of_batches) as progress_bar:
        progress_bar.update(0)
    # print("running epoch")
        while True:
            batch, target = dataset.getData()
            # batch.to(device)
            # target.to(device)
            if batch == None: break;
            # print("batch " + str(counter) + " of " + str(number_of_batches))
            optimizer.zero_grad()
            states = model.detach_states(states)
            output, states = model(batch, states)
            # print(list(output))
            # print(list(target))
            loss = loss_func(output, target)
            # print(loss)
            # print()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            counter += 1
            progress_bar.update(counter)
    meanLoss = np.mean(losses)
    print("Loss:",meanLoss)
    return model, meanLoss
