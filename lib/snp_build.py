# SNP scanner and pickler

# STATUS: Completed
# Run-time: 1 hour and 31 mins before market opens

#***          Start ib_insync (run once)       *****
#___________________________________________________

from ib_insync import *
# util.startLoop()
ib = IB().connect('127.0.0.1', 1300, clientId=2)

import os
from snplist import fspath, snp_list, get_opt
import datetime

# declaration
hrs = 2   # approximate time taken to get options for all scrips

# qualified list of stocks and index, with their contracts and chains
symconchain = snp_list(ib)

symbols = symconchain[0] # all symbols

contracts = symconchain[1] # all contracts

chains = symconchain[2] # all chains

# Take only pickle files. Remove directories and files starting with underscore (for underlyings)
fs = [f for f in os.listdir(fspath) if (f[-7:] == 'opt.pkl')] # list of opt pickle files
xs = [f[:-8] for f in fs] # symbols for existing pickle files with options

if fs: # if the file list is not empty

    # Get modified time, fail time and identify where the scrip has failed
    fsmod = {f: os.path.getmtime(fspath + '/' + f) for f in fs}
    failtime = max([v for k, v in fsmod.items()])
    failscrip = [k[:-4] for k, v in fsmod.items() if v == failtime][0]
    
    # now - porgram runtime
    floortime = (datetime.datetime.now() - datetime.timedelta(hours = hrs)).timestamp()  

    if failtime < floortime:   # program failed to fully pickle
        restartfrom = 0  # restart from zero
    else:
        restartfrom = symbols.index(failscrip[:-4]) + 1 # restart from where it failed

else: 
    restartfrom = 0  # restart from zero if the file list is empty

[get_opt(ib, contract, chain) for contract, chain in zip(contracts[restartfrom:], chains[restartfrom:])]
print('SNP build completed!')
ib.disconnect()