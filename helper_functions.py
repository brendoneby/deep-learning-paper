import itertools
import numpy as np
from BSE import *

def calcPstar(tape):
    prices = []
    for trade in tape:
        if trade['type'] == 'Trade':
            prices.append(trade['price'])
    prices = np.array(prices)
    n = len(prices)
    lda = .95
    w = np.array([lda**(n-i) for i in range(n)])
    pstar = np.sum(prices*w)/np.sum(w)
    alpha = np.sqrt( np.sum([(prices[i] - pstar)**2 for i in range(n)])/n )/pstar
    return pstar, alpha

def getSnapshot(lob, time, order, trade, cust_order, prev_trade_time):
    # print(lob)
    # print(order)
    # print(trade)
    pb = lob['bids']['worst'] if lob['bids']['best'] == None else lob['bids']['best'] # best bid
    qa = 0 if len(lob['asks']['lob']) == 0 else lob['asks']['lob'][0][1] # quantity for best ask
    pa = lob['asks']['worst'] if lob['asks']['best'] == None else lob['asks']['best'] # best ask
    qb = 0 if len(lob['bids']['lob']) == 0 else lob['bids']['lob'][-1][1] # quantity for best bid
    tqa = lob['asks']['n']
    tqb = lob['bids']['n']
    microprice = 0 if (qa+qb) == 0 else ((pb*qa) + (pa*qb))/(qa+qb)
    p_star, s_alpha = calcPstar(lob['tape'])
    snapshot_dict = {
        'time': time,
        'flag': 1 if trade['best'] == 1 else 0,  # 1 for hit/lift
        'customerPrice': cust_order,
        'bid_ask_spread': abs(pa - pb), # difference between best ask and best bid
        'midprice': ((pb + pa) / 2), # average of best ask and best bid
        'microprice': microprice,
        'best_bid_price': pb,
        'best_ask_price': pa,
        'time_since_prev_trade': 0 if prev_trade_time == 0 else time - prev_trade_time,
        'LOB_imbalance': (tqb - tqa) / (tqb + tqa),
        'total_quotes': tqa + tqb,
        'p_star': p_star,
        'smiths_alpha': s_alpha,
        'trade_price': trade['price']
    }
    
    return snapshot_dict.values()

def normalize(snapshot):
    #MASUM: normalize snapshots
    return snapshot


# def run_sessions(buyers_spec, sellers_spec, n_trials):
#     # set up common parameters for all market sessions
#     start_time = 0.0
#     end_time = 600.0
#     duration = end_time - start_time

#     # The code below sets up symmetric supply and demand curves at prices from 50 to 150, P0=100
#     range1 = (50, 150)
#     supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]
#     range2 = (50, 150)
#     demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

#     order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
#                    'interval': 30, 'timemode': 'drip-poisson'}

#     traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

#     # run a sequence of trials, one session per trial
#     verbose = True
#     tdump = open('avg_balance.csv', 'w')
#     trial = 1

#     while trial < (n_trials + 1):
#         trial_id = 'sess%04d' % trial
#         dump_all = True

#         market_session(trial_id, start_time, end_time, traders_spec, order_sched, tdump, dump_all, verbose)
#         tdump.flush()
#         trial = trial + 1

#     tdump.close()

def get_all_trader_configs():
    avail_traders = ['AA','GDX', 'GVWY','ZIC','SHVR','SNPR','ZIP']
    avail_configs = [(20, 10, 5, 5), (10, 10, 10, 10), (15, 10, 10, 5), (15, 15, 5, 5), (25, 5, 5, 5)]
    trader_perms = set(itertools.combinations(avail_traders,4))
    config_perms = set()
    for config in avail_configs:
        config_perms |= set(itertools.permutations(config))

    perms = list()
    for trader_set in trader_perms:
        # If we don't wanna include permutations where AA and ZIP don't occur
        # or add 
        # if 'AA' not in trader_set and 'ZIP' not in trader_set: continue
        for config_set in config_perms:
            perms.append(list(zip(trader_set,config_set)))
    print(perms[0])
    print(len(perms))
    return perms