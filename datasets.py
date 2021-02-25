import numpy as np
from numpy import genfromtxt
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append("/Users/davinci/NU_work/Advanced Deep/deep-learning-paper/")
# print(sys.path)

# Only for Data preparation
def merge_csv_files():
    # This function merges different csv files generated on different machines
    fout=open("data/snapshots.csv","a")
    for num in range(1,5):
        for line in open("data/snapshots_train"+str(num)+".csv"):
            fout.write(line)
    fout.close()

def load_text_data(path: str, type: str):
    """
    read csv file
    :param path:
    :return:
    """
    
    train = genfromtxt(path, delimiter=',')
    val = train[:100000]
    train = train[100000:]
    
    return train, val

class Sequence_Dataset(Dataset):
    def __init__(self, x:torch.LongTensor, y:torch.LongTensor):
        self.x = x
        self.y = y
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

def _build_dataloader(data:np.ndarray, batch_size:int) -> DataLoader:
    """
    :param data: input array of floats
    :param batch_size: hyper parameter, for mini-batch size
    :return: DataLoader for SGD
    """
    
    # cut off any data that will create incomplete batches
    num_batches = data.shape[0] // batch_size
    data = data[:num_batches*batch_size]
    inputs = data[:,13]
    targets = data[:,:13]
    
    # create Dataset object and from it create data loader
    dataset = Sequence_Dataset(x=inputs, y=targets)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)