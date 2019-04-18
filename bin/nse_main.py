# nse_main.py

from ib_insync import *

from os import listdir

from helper import get_connected, get_nses, get_nse_options, catch

keep_pickles = False   # keep already pickled symbols

with get_connected('nse', 'live') as ib: 
    
    fspath = '../data/nse/' # path for pickles
    
    # get the list of underlying contracts and dictionary of lots
    qlm = get_nses(ib)
    undContracts = qlm[0]
    lots_dict = qlm[1]
    margins_dict = qlm[2]

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
        lotsizes = [lots_dict[s] for s in symbols]
        margins = [margins_dict[s] for s in symbols]
        [catch(lambda: get_nse_options(ib, u, p, z, m)) for u, p, z, m in zip(contracts, prices, lotsizes, margins)]
    else:
        [catch(lambda: get_nse_options(ib, u, p, z, m)) for u, p, z, m in zip(undContracts, undPrices.values(), lots_dict.values(), margins_dict.values())]