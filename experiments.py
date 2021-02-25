from DeepTrader import *
from data_functions import *

def train_deep_model():
    # TODO: load data, and pre-process
    dataloader = None
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