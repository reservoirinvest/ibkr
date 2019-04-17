# snp_main.py

from ib_insync import *

from os import listdir

from helper import get_snps, get_snp_options, catch

fspath = '../data/snp/' # path for pickles
keep_pickles = True   # keep already pickled symbols

def get_connected(market, trade_type):
    ''' get connected to ibkr
    Args: 
       (market) as string <'nse'> | <'snp'>
       (trade_type) as string <'live'> | <'paper'>
    Returns:
        (ib) object if successful
    '''
    
    ip = (market.upper(), trade_type.upper())
    
    #host dictionary
    hostdict = {('NSE', 'LIVE'): 3000,
                ('NSE', 'PAPER'): 3001,
                ('SNP', 'LIVE'): 1300,
                ('SNP', 'PAPER'): 1301,}
    
    host = hostdict[ip]
    
    cid = 1 # initialize clientId
    max_cid = 5 # maximum clientId allowed. max possible is 32

    for i in range(cid, max_cid):
        try:
            ib = IB().connect('127.0.0.1', host, clientId=i)
            
        except Exception as e:
            print(e) # print the error
            continue # go to next
            
        break # successful try
        
    return ib

# get connected
ib = get_connected('snp', 'live')

try:
    ib.isConnected()
except Exception as e:
    ib = get_connected('snp', 'live')
    
if not ib.isConnected():
    ib = get_connected('snp', 'live')

# get all the underlying contracts with prices
undContracts = get_snps(ib)
tickers = ib.reqTickers(*undContracts)
undPrices = {t.contract.symbol: t.marketPrice() for t in tickers} # {symbol: undPrice}

util.logToFile(fspath+'_errors.log')  # create log file
with open(fspath+'_errors.log', 'w'): # clear the previous log
    pass

fs = listdir(fspath)
optsList = [f[:-4] for f in fs if f[-3:] == 'pkl']

if keep_pickles:
    contracts = [m for m in undContracts if m.symbol not in optsList]
    symbols = [c.symbol for c in contracts]
    prices = [undPrices[s] for s in symbols]
    [catch(lambda: get_snp_options(ib, b, c)) for b, c in zip(contracts, prices)]
else:
    [catch(lambda: get_snp_options(ib, b, c)) for b, c in zip(undContracts, undPrices.values())]