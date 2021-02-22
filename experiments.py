from DeepTrader import *
from helper_functions import *

def train_deep_model():
    # TODO: load data, and pre-process
    dataloader = None
    num_epochs = 20
    train(num_epochs, dataloader)

def generate_data():
    n_trials = 2
    all_configs = get_all_trander_configs()
    trader_spec = all_configs[0]
    run_sessions(trader_spec, trader_spec, n_trials)

def run_experiment():
    pass