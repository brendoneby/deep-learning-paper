import itertools
import numpy as np

maxVals = []
minVals = []

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

def getSnapshot(lob, time, order=None, trade=None, cust_order=0, prev_trade_time=None, isAsk = None):
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

    if trade is not None: isAsk = 1 if pa == trade['price'] else 0  # 1 for ask, 0 for bid
    if prev_trade_time is None: prev_trade_time = getLastTrade(lob)

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
        'LOB_imbalance': (tqb - tqa) / (tqb + tqa),
        'total_quotes': tqa + tqb,
        'p_star': p_star,
        'smiths_alpha': s_alpha,
        'trade_price': trade['price'] if trade is not None else None
    }
    
    return snapshot_dict.values()

def getLastTrade(lob):
    tape = lob['tape']
    for i in range(1,len(tape)):
        tapeItem = tape[-1*i]
        if tapeItem['type'] == 'Trade':
            return tapeItem['time']
    return 0

def normalize(min_val, max_val, row):
    
    #MASUM: normalize snapshots
    return snapshot