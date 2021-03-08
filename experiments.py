import os
from fnmatch import fnmatch

from DeepTrader import *
from data_functions import *
from datasets import *

def train_deep_model():
    data = load_data("data/snapshots.csv", "train")
    batch_size = 16384
    dataloader = build_dataloader(data, batch_size)
    num_epochs = 20
    train(num_epochs, dataloader)

def generate_data():
    n_trials = 2
    all_configs = get_all_trader_configs()
    i = 0
    for config in all_configs:
        i+=1
        print("trial", i)
        run_sessions(config, config, n_trials)

def process_lobster(folder_root, output_fn):
    pattern = "*_message_10.csv"
    i = 0
    for path, subdirs, files in os.walk(folder_root):
        for name in files:
            if fnmatch(name, pattern):
                fullname = os.path.join(path, name)
                base = fullname[0:fullname.index('_message')]
                print("parsing "+str(i)+" of "+str(len(files)/2))
                parse_lobster_data(base, output_fn)
                i += 1

def run_OMT():
    trader = "DEEP"
    avail_traders = get_avail_traders()
    for opponent in avail_traders:
        if opponent == 'AA': continue
        one_in_many_test(trader, opponent)

def run_BGT(avail_traders = None):
    trader = "DEEP"
    if avail_traders is None:
        avail_traders = get_avail_traders()
    for opponent in avail_traders:
        balanced_group_test(trader, opponent)

def run_tests(avail_traders = None, trader = "DEEP"):
    if avail_traders is None:
        avail_traders = get_avail_traders()
    for opponent in avail_traders:
        balanced_group_test(trader, opponent)
        one_in_many_test(trader, opponent)

# run_tests(['AA','GVWY','ZIC','SHVR','SNPR','ZIP'])
run_tests(['GDX','AA','GVWY','ZIC','SHVR','SNPR','ZIP'], 'RDP')

# generate_bgt_plot("DEEP", "AA")
# generate_data()
# process_lobster("/Users/brendoneby/Downloads/_data_dwn_13_359__JNJ_2019-12-31_2020-12-31_10", "JNJ")