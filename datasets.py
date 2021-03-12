import numpy as np
from numpy import genfromtxt
import torch
import time
from itertools import islice
from helper_functions import getSnapshot

def noramlize_training_data(data):
    data[:,1] = np.sign(data[:,13] - data[:,6])
    max_vals = np.max(data,axis=0)
    min_vals = np.min(data,axis=0)
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
    return genfromtxt(path, delimiter=',')

def parse_lobster_data(fn, output_fn):
    print("loading data from "+fn)
    messages = genfromtxt(fn+"_message_10.csv", delimiter=',')
    orderbook = genfromtxt(fn+"_orderbook_10.csv", delimiter=',')
    tape = []
    snapshots = []
    prev_trade_time = 0
    t0 = time.time()
    for i in range(len(messages)):
        message = messages[i]
        message_type = message[1]
        if(message_type == 4 or message_type == 5):
            trade_time = message[0]
            qty = message[3]
            price = message[4]/10000
            trade = {}
            trade['time'] = trade_time
            trade['type'] = 'Trade'
            trade['price'] = price
            trade['qty'] = qty

            isAsk = 1 if message[5] == 1 else 0

            lob_index = max(0,i-1)
            lob = getLobsterLob(orderbook[lob_index], tape)

            snapshot = getSnapshot(lob, trade_time, trade=trade, prev_trade_time=prev_trade_time, isAsk=isAsk)
            snapshots.append(list(snapshot))

            tape.append(trade)
            prev_trade_time = trade_time
    t1 = time.time()
    print("snapshots took: " + str(t1-t0))

    print("normalizing...")
    norm_snapshots = noramlize_lobster_data(np.array(snapshots))
    t2 = time.time()
    print("normalizing took: " + str(t2-t1))
    fout=open("data/lobster_snapshots"+output_fn+".csv","a")
    for snapshot in norm_snapshots:
        str_snapshot = [str(el) for el in snapshot]
        fout.write(",".join(str_snapshot)+"\n")
    fout.close()
    t3 = time.time()
    print("writing took: " + str(t3-t2))
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
    def __init__(self, fn, batch_size, device):
        self.fn = fn
        self.infile = None
        self.reset_infile()
        self.batch_size = batch_size
        self.len = 10396317
        self.device = device
        print("len:",self.len)

    def reset_infile(self):
        if self.infile != None: self.infile.close()
        self.infile = open(self.fn, 'r')

    def getData(self):
        gen = islice(self.infile, self.batch_size)
        data = np.genfromtxt(gen, delimiter=',')
        if data.shape[0] < self.batch_size:
            self.reset_infile()
            return None, None
        inputs = torch.tensor(data[:,:13],dtype=torch.float, device=self.device)
        inputs = inputs.reshape(1,inputs.shape[0],inputs.shape[1])
        targets = torch.tensor(data[:,13],dtype=torch.float, device=self.device)
        targets = targets.reshape(1,targets.shape[0],1)
        return inputs, targets

    def __len__(self):
        return self.len

def build_dataloader(fn, batch_size, device):
    """
    :param data: input array of floats
    :param batch_size: hyper parameter, for mini-batch size
    :return: DataLoader for SGD
    """
    dataset = Sequence_Dataset(fn, batch_size, device)
    return dataset
