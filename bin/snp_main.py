# snp_main.py

from ib_insync import *

from os import listdir

from helper import get_snps, get_snp_options, catch, get_connected

keep_pickles = False   # keep already pickled symbols

with get_connected('snp', 'live') as ib:
    
    fspath = '../data/snp/' # path for pickles

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