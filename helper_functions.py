import itertools
import numpy as np

max_vals = np.array([6.00000000e+02, 1.00000000e+00, 1.50000000e+02, 9.50000000e+02,
 5.75000000e+02, 1.00000000e+03, 1.50000000e+02, 1.00000000e+03,
 5.27125000e+01, 1.00000000e+00, 7.40000000e+01, 1.50000000e+02,
 5.60316505e-01, 1.50000000e+02])
min_vals = np.array([ 1.62500000e-01, 0.00000000e+00, 5.00000000e+01, 1.42108547e-14,
  2.50000000e+01, 1.00000000e+00, 0.00000000e+00, 5.00000000e+01,
  0.00000000e+00, -1.00000000e+00, 1.00000000e+00, 5.00000000e+01,
  0.00000000e+00, 5.00000000e+01])

def calcPstar(tape):
    prices = []
    for trade in tape:
        if trade['type'] == 'Trade':
            prices.append(trade['price'])
    prices = np.array(prices)
    n = len(prices)
    if n == 0: return 0,0   #No Trades Yet
    lda = .95
    w = np.array([lda**(n-i) for i in range(n)])
    pstar = np.sum(prices*w)/np.sum(w)
    alpha = np.sqrt( np.sum([(prices[i] - pstar)**2 for i in range(n)])/n )/pstar
    return pstar, alpha

def getSnapshot(lob, time, order=None, trade=None, cust_order=0, prev_trade_time=None, isAsk = None):
    pb = lob['bids']['worst'] if lob['bids']['best'] == None else lob['bids']['best'] # best bid
    qa = 0 if len(lob['asks']['lob']) == 0 else lob['asks']['lob'][0][1] # quantity for best ask
    pa = lob['asks']['worst'] if lob['asks']['best'] == None else lob['asks']['best'] # best ask
    qb = 0 if len(lob['bids']['lob']) == 0 else lob['bids']['lob'][-1][1] # quantity for best bid
    tqa = lob['asks']['n']
    tqb = lob['bids']['n']
    microprice = 0 if (qa+qb) == 0 else ((pb*qa) + (pa*qb))/(qa+qb)
    p_star, s_alpha = calcPstar(lob['tape'])

    if isAsk is None and trade is not None: isAsk = 1 if pa == trade['price'] else 0  # 1 for ask, 0 for bid
    if prev_trade_time is None: prev_trade_time = getLastTrade(lob)

    trade_price = trade['price'] if trade is not None else -1
    total_quotes = (tqb + tqa)
    lob_imbalance = 0 if total_quotes == 0 else (tqb - tqa) / total_quotes

    snapshot_dict = {
        'time': time,
        'flag': isAsk,
        'customerPrice': cust_order,
        'bid_ask_spread': abs(pa - pb), # difference between best ask and best bid
        'midprice': ((pb + pa) / 2), # average of best ask and best bid
        'microprice': microprice,
        'best_bid_price': pb,
        'best_ask_price': pa,
        'time_since_prev_trade': 0 if prev_trade_time == 0 else time - prev_trade_time,
        'LOB_imbalance': lob_imbalance,
        'total_quotes': total_quotes,
        'p_star': p_star,
        'smiths_alpha': s_alpha,
        'trade_price': trade_price
    }
    
    return snapshot_dict.values()

def getLastTrade(lob):
    tape = lob['tape']
    for i in range(1,len(tape)):
        tapeItem = tape[-1*i]
        if tapeItem['type'] == 'Trade':
            return tapeItem['time']
    return 0

def normalize(data):
    max_vals_l = max_vals[:len(data)]
    min_vals_l = min_vals[:len(data)]
    return (data-min_vals_l)/(max_vals_l-min_vals_l)

def unnormalizePrice(price):
    maxPrice = max_vals[-1]
    minPrice = min_vals[-1]
    real_price = price * (maxPrice-minPrice) + minPrice
    return real_price

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
    return perms

def get_avail_traders():
    return ['AA','GDX', 'GVWY','ZIC','SHVR','SNPR','ZIP']