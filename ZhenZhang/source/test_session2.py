import itertools

from BSE import *

def run_sessions(buyers_spec, sellers_spec, n_trials):

    # set up common parameters for all market sessions
    start_time = 0.0
    end_time = 600.0
    duration = end_time - start_time

    # The code below sets up symmetric supply and demand curves at prices from 50 to 150, P0=100

    range1 = (50, 150)
    supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}
                       ]

    range2 = (50, 150)
    demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}
                       ]

    order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
                   'interval': 30, 'timemode': 'drip-poisson'}
    # Use 'periodic' if you want the traders' assignments to all arrive simultaneously & periodically
    #               'interval': 30, 'timemode': 'periodic'}

    traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

    # run a sequence of trials, one session per trial

    verbose = True
    tdump = open('avg_balance.csv', 'w')
    trial = 1

    while trial < (n_trials + 1):
        trial_id = 'sess%04d' % trial
        dump_all = True

        market_session(trial_id, start_time, end_time, traders_spec, order_sched, tdump, dump_all, verbose)
        tdump.flush()
        trial = trial + 1

    tdump.close()

def get_all_trander_configs():
    avail_traders = ['GVWY','ZIC','SHVR','SNPR','ZIP','AA','GDX']
    avail_configs = [(20, 10, 5, 5), (10, 10, 10, 10), (15, 10, 10, 5), (15, 15, 5, 5), (25, 5, 5, 5)]
    trader_perms = set(itertools.combinations(avail_traders,4))
    config_perms = set()
    for config in avail_configs:
        config_perms |= set(itertools.permutations(config))

    perms = list()
    for trader_set in trader_perms:
        for config_set in config_perms:
            perms.append(zip(trader_set,config_set))
    print(perms[0])
    print(len(perms))
    return perms


n_trials = 2
all_configs = get_all_trander_configs()
trader_spec = all_configs[0]
run_sessions(trader_spec, trader_spec, n_trials)