import numpy as np
from numpy import genfromtxt
import torch
from torch.functional import norm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import sys
from itertools import islice

# sys.path.append("/Users/davinci/NU_work/Advanced Deep/deep-learning-paper/")
# print(sys.path)

def noramlize_training_data(data):
    data[:,1] = np.sign(data[:,13] - data[:,6])
    max_vals = np.max(data,axis=0)
    min_vals = np.min(data,axis=0)
    norm_data = (data-min_vals)/(max_vals-min_vals)
    np.savetxt("data/snapshots_normalized.csv", norm_data, delimiter=",")

# Only for Data preparation
def merge_csv_files():
    # This function merges different csv files generated on different machines
    fout=open("data/snapshots.csv","a")
    for num in range(1,5):
        for line in open("data/snapshots_train"+str(num)+".csv"):
            fout.write(line)
    fout.close()

def load_data(path: str, type: str):
    """
    read csv file
    :param path:
    :return:
    """

    train = genfromtxt(path, delimiter=',')
    # val = train[:100000]
    # train = train[100000:]
    
    return train

class Sequence_Dataset(Dataset):
    # def __init__(self, x:torch.LongTensor, y:torch.LongTensor):
    def __init__(self, fn, batch_size):
        # self.x = x
        # self.y = y
        # self.len = x.shape[0]
        self.fn = fn
        self.infile = None
        self.reset_infile()
        self.batch_size = batch_size
        self.len = 10396317
        # self.len = 0
        # with open(fn, 'r') as infile:
        #     while True:
        #         gen = islice(infile, batch_size)
        #         # print(gen)
        #         l = len(tuple(gen))
        #         self.len += l
        #         if l < batch_size:
        #             break
        print("len:",self.len)

    def reset_infile(self):
        if self.infile != None: self.infile.close()
        self.infile = open(self.fn, 'r')

    def __getitem__(self, idx):
        # print("Loading batch ")
        gen = islice(self.infile, self.batch_size)
        data = np.genfromtxt(gen, delimiter=',')
        # print(data.shape)
        # print(data)
        if data.shape[0] < self.batch_size:
            self.reset_infile()
            return [], []
        inputs = data[:,13]
        targets = data[:,:13]
        # print(inputs)
        # print(targets)
        return inputs, targets

    def __len__(self):
        return self.len

# def build_dataloader(data:np.ndarray, batch_size:int) -> DataLoader:
#     """
#     :param data: input array of floats
#     :param batch_size: hyper parameter, for mini-batch size
#     :return: DataLoader for SGD
#     """
#
#     # cut off any data that will create incomplete batches
#     num_batches = data.shape[0] // batch_size
#     data = data[:num_batches*batch_size]
#     inputs = data[:,13]
#     targets = data[:,:13]
#
#     # create Dataset object and from it create data loader
#     dataset = Sequence_Dataset(x=inputs, y=targets)
#     return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

def build_dataloader(fn, batch_size) -> DataLoader:
    """
    :param data: input array of floats
    :param batch_size: hyper parameter, for mini-batch size
    :return: DataLoader for SGD
    """

    # create Dataset object and from it create data loader
    dataset = Sequence_Dataset(fn, batch_size)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# data = load_data("data/snapshots.csv", "train")
#
# noramlize_training_data(data)