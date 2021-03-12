import sys
import os
from fnmatch import fnmatch
from DeepTrader import *
from helper_functions import *
from tests import *
from sessions import *
from datasets import *

def train_deep_model():
    data = load_data("data/snapshots.csv", "train")
    batch_size = 16384
    dataloader = build_dataloader(data, batch_size)
    num_epochs = 20
    train(num_epochs, dataloader)

def generate_data(n_trials = None):
    if n_trials is None: n_trials = 1
    all_configs = get_all_trader_configs()
    i = 0
    for config in all_configs:
        i+=1
        print("trial", i)
        run_sessions(config, config, n_trials)

def normalize_data(fn = None):
    if not fn: fn = "data/snapshots.csv"
    data = load_data(fn, "train")
    noramlize_training_data(data)

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

def run_OMT(trader = None):
    if not trader: trader = "DEEP"
    avail_traders = get_avail_traders()
    for opponent in avail_traders:
        if opponent == 'AA': continue
        one_in_many_test(trader, opponent)

def run_BGT(avail_traders = None, trader = None):
    if not trader: trader = "DEEP"
    if avail_traders is None:
        avail_traders = get_avail_traders()
    for opponent in avail_traders:
        balanced_group_test(trader, opponent)

def run_tests(avail_traders = None, trader = None):
    if not trader: trader = "DEEP"
    if avail_traders is None:
        avail_traders = get_avail_traders()
    for opponent in avail_traders:
        balanced_group_test(trader, opponent)
        one_in_many_test(trader, opponent)

if __name__ == "__main__":
    function = sys.argv[1] if len(sys.argv) > 1 else "test"
    arg1 = sys.argv[2] if len(sys.argv) > 2 else None
    arg2 = sys.argv[3] if len(sys.argv) > 3 else None

    if function == 'test': run_tests(trader=arg1)
    if function == 'generateData': generate_data(n_trials=int(arg1))
    if function == 'normalize': normalize_data(fn=arg1)
    if function == 'processLobster': process_lobster(folder_root=arg1, output_fn=arg2)