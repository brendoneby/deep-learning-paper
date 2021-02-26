from DeepTrader import *
from data_functions import *

def train_deep_model():
    # TODO: load data, and pre-process
    dataloader = None
    num_epochs = 20
    train(num_epochs, dataloader)

def generate_data():
<<<<<<< Updated upstream
    n_trials = 12
    all_configs = get_all_trader_configs()
    i = 0
    for config in all_configs:
        i+=1
        print("trial", i)
        run_sessions(config, config, n_trials)
=======
    n_trials = 4
    all_configs = get_all_trader_configs()
    i = 0;
    for config in all_configs:
        print("trial", i)
        run_sessions(config, config, n_trials)
        i+=1
>>>>>>> Stashed changes

def run_experiment():
    #TODO: build
    pass

generate_data()