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
        dump_all = True

        market_session(trial_id, start_time, end_time, traders_spec, order_sched, tdump, dump_all, verbose, total_dump)
        tdump.flush()
        trial = trial + 1

    tdump.close()
