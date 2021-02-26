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
    n_trials = 12
    all_configs = get_all_trader_configs()
    i = 0
    for config in all_configs:
        i+=1
        print("trial", i)
        run_sessions(config, config, n_trials)

def run_experiment():
    #TODO: build
    pass

generate_data()