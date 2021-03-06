import numpy as np
from numpy import genfromtxt
import torch
from torch.functional import norm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import sys
from itertools import islice
from helper_functions import getSnapshot

# sys.path.append("/Users/davinci/NU_work/Advanced Deep/deep-learning-paper/")
# print(sys.path)

def noramlize_training_data(data):
    data[:,1] = np.sign(data[:,13] - data[:,6])
    max_vals = np.max(data,axis=0)
    min_vals = np.min(data,axis=0)
    # print(max_vals)
    # print(min_vals)
    norm_data = (data-min_vals)/(max_vals-min_vals)
    np.savetxt("data/snapshots_normalized.csv", norm_data, delimiter=",")

def noramlize_lobster_data(data):
    max_vals = np.max(data,axis=0)
    min_vals = np.min(data,axis=0)
    norm_data = (data-min_vals)/(max_vals-min_vals)
    norm_data[:,2] = 0    #set customer price to 0, since we dont have that information in lobster
    return norm_data

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

def parse_lobster_data(fn, output_fn):
    print("loading data from "+fn)
    messages = genfromtxt(fn+"_message_10.csv", delimiter=',')
    orderbook = genfromtxt(fn+"_orderbook_10.csv", delimiter=',')
    tape = []
    snapshots = []
    prev_trade_time = 0
    lob = None
    for i in range(len(messages)):
        message = messages[i]
        message_type = message[1]
        if(message_type == 4 or message_type == 5):
            time = message[0]
            qty = message[3]
            price = message[4]/10000
            trade = {}
            trade['time'] = time
            trade['type'] = 'Trade'
            trade['price'] = price
            trade['qty'] = qty

            isAsk = 1 if message[5] == 1 else 0

            # lob_row = orderbook[i]
            if lob is None: lob = getLobsterLob(orderbook[i], tape)

            snapshot = getSnapshot(lob, time, trade=trade, prev_trade_time=prev_trade_time, isAsk=isAsk)
            snapshots.append(list(snapshot))
            # print(message)
            # # print(lob_row)
            # print(trade)
            # print(lob)
            # print(snapshot)

            tape.append(trade)
            prev_trade_time = time
            # assert(False)
        lob_row = orderbook[i]
        lob = getLobsterLob(lob_row, tape)

    norm_snapshots = noramlize_lobster_data(np.array(snapshots))
    print(norm_snapshots)
    print(norm_snapshots.shape)
    fout=open("data/lobster_snapshots"+output_fn+".csv","a")
    for snapshot in norm_snapshots:
        str_snapshot = [str(el) for el in snapshot]
        fout.write(",".join(str_snapshot)+"\n")
    fout.close()
    return snapshots


def getLobsterLob(lob_row, tape):
    lob = {}
    lob['tape'] = tape
    lob['bids'] = {}
    lob['bids']['lob'] = []
    lob['bids']['n'] = 0
    lob['asks'] = {}
    lob['asks']['lob'] = []
    lob['asks']['n'] = 0
    for i in range(0, len(lob_row), 4):
        ask_price = lob_row[i] / 10000
        ask_qty = lob_row[i + 1]
        lob['asks']['lob'].append([ask_price, ask_qty])
        lob['asks']['n'] += ask_qty

        bid_price = lob_row[i + 2] / 10000
        bid_qty = lob_row[i + 3]
        lob['bids']['lob'].insert(0, [bid_price, bid_qty])
        lob['bids']['n'] += bid_qty

        if (i == 0):
            lob['asks']['best'] = ask_price
            lob['bids']['best'] = bid_price
        if (i + 4 == len(lob_row)):
            lob['asks']['worst'] = ask_price
            lob['bids']['worst'] = bid_price
    return lob

class Sequence_Dataset():
    # def __init__(self, x:torch.LongTensor, y:torch.LongTensor):
    def __init__(self, fn, batch_size, device):
        # self.x = x
        # self.y = y
        # self.len = x.shape[0]
        self.fn = fn
        self.infile = None
        self.reset_infile()
        self.batch_size = batch_size
        self.len = 10396317
        self.device = device
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

    def getData(self):
        # print("Loading batch ")
        gen = islice(self.infile, self.batch_size)
        data = np.genfromtxt(gen, delimiter=',')
        # print(data.shape)
        # print(data)
        if data.shape[0] < self.batch_size:
            self.reset_infile()
            return None, None
        inputs = torch.tensor(data[:,:13],dtype=torch.float, device=self.device)
        inputs = inputs.reshape(1,inputs.shape[0],inputs.shape[1])
        targets = torch.tensor(data[:,13],dtype=torch.float, device=self.device)
        targets = targets.reshape(1,targets.shape[0],1)
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

def build_dataloader(fn, batch_size, device) -> DataLoader:
    """
    :param data: input array of floats
    :param batch_size: hyper parameter, for mini-batch size
    :return: DataLoader for SGD
    """

    # create Dataset object and from it create data loader
    dataset = Sequence_Dataset(fn, batch_size, device)
    return dataset
    # return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

# data = load_data("data/snapshots.csv", "train")
# #
# noramlize_training_data(data)

# merge_csv_files()

# fn = "/Users/brendoneby/Downloads/_data_dwn_13_359__JNJ_2019-12-31_2020-12-31_10/JNJ_2019-12-31_34200000_57600000"
# snapshots = parse_lobster_data(fn)
# print(snapshots)