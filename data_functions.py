import itertools
import pandas
import matplotlib.pyplot as plt
from BSE import market_session


def run_sessions(buyers_spec, sellers_spec, n_trials, recordSnapshots = True, balance_file = 'avg_balance.csv', total_file = 'total_balance.csv'):
    # set up common parameters for all market sessions
    start_time = 0.0
    end_time = 600.0
    duration = end_time - start_time

    # The code below sets up symmetric supply and demand curves at prices from 50 to 150, P0=100
    range1 = (50, 150)
    supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]
    range2 = (50, 150)
    demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

    order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
                   'interval': 30, 'timemode': 'drip-poisson'}

    traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

    # run a sequence of trials, one session per trial
    verbose = True
    tdump = open(balance_file, 'w')
    total_dump = open(total_file, 'w')
    trial = 1

    while trial < (n_trials + 1):
        trial_id = 'sess%04d' % trial
        dump_all = False

        market_session(trial_id, start_time, end_time, traders_spec, order_sched, tdump, dump_all, verbose, total_dump)
        tdump.flush()
        trial = trial + 1

    tdump.close()

def get_all_trader_configs():
    avail_traders = get_avail_traders()
    avail_configs = [(20, 10, 5, 5), (10, 10, 10, 10), (15, 10, 10, 5), (15, 15, 5, 5), (25, 5, 5, 5)]
    trader_perms = set(itertools.combinations(avail_traders,4))
    config_perms = set()
    for config in avail_configs:
        config_perms |= set(itertools.permutations(config))

    perms = list()
    for trader_set in trader_perms:
        # if 'AA' not in trader_set and 'ZIP' not in trader_set: continue
        for config_set in config_perms:
            perms.append(list(zip(trader_set,config_set)))
    print(perms[0])
    print(len(perms))
    return perms

def get_avail_traders():
    return ['AA','GDX', 'GVWY','ZIC','SHVR','SNPR','ZIP']

def get_test_file(type, traderOne, traderMany):
    return 'total_balance_{0}_{1}_{2}.csv'.format(type, traderOne, traderMany)

def parse_test_file(file):
    data = pandas.read_csv(file).to_numpy()
    perf = {}
    for row in data:
        for col in range(4,len(row),4):
            item = row[col:col+4]
            if isinstance(item[0],str):
                trader = item[0].strip()
                if trader not in perf: perf[trader] = []
                perf[trader].append(item[3])
    return perf

def generate_box_plots(file, title):
    data = parse_test_file(file)
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title(title)
    ax.boxplot(data.values(), whis=1.5)
    ax.set_xticklabels(data.keys())
    plt.show()

def one_in_many_test(traderOne, traderMany):
    test_type = "OMT"
    traders_spec = [(traderOne,1),(traderMany,39)]
    total_file = get_test_file(test_type, traderOne, traderMany)

    n_trials = 100
    print("Starting One-vs-many test for "+traderOne+" vs "+traderMany)
    run_sessions(traders_spec, traders_spec, n_trials, total_file=total_file)
    generate_omt_plot(traderOne, traderMany)

def generate_omt_plot(traderOne, traderMany):
    file = get_test_file("OMT", traderOne, traderMany)
    title = "One-in-many: "+traderOne + " vs " + traderMany
    generate_box_plots(file, title)

def balanced_group_test(traderOne, traderTwo):
    test_type = "BGT"
    traders_spec = [(traderOne,20),(traderTwo,20)]
    total_file = get_test_file(test_type, traderOne, traderTwo)
    n_trials = 100
    print("Starting Balanced Group test for "+traderOne+" vs "+traderTwo)
    run_sessions(traders_spec, traders_spec, n_trials, total_file=total_file)
    generate_bgt_plot(traderOne, traderTwo)

def generate_bgt_plot(traderOne, traderTwo):
    file = get_test_file("BGT", traderOne, traderTwo)
    title = "Balanced-group-test: "+traderOne + " vs " + traderTwo
    generate_box_plots(file, title)
