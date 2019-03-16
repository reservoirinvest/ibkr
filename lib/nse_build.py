# NSE scanner and pickler

# STATUS: WIP
# Run-time: 

#***          Start ib_insync (run once)       *****
#_______________________________________________

from ib_insync import *
# util.startLoop()
ib = IB().connect('127.0.0.1', 3000, clientId=2)

import os
from nselist import fspath, nse_list, get_opt
import datetime

# declaration
hrs = 2   # approximate time taken to get options for all scrips

# qualified list of stocks and index, with their contracts and chains
df_nselist = nse_list(ib)

# make the dfs from nselist
dfs = [df_nselist[df_nselist.symbol == s].reset_index(drop=True) for s in list(df_nselist.symbol.unique())]

# Take only pickle files. Remove directories and files starting with underscore (for underlyings)
fs = [f for f in os.listdir(fspath) if (f[-7:] == 'opt.pkl')] # list of opt pickle files

all_pickles = [fspath+f for f in os.listdir(fspath) if f.endswith('.pkl')]  # all pickled files
av_pkl_symbols = [f[:-8] for f in fs] # available pickle symbols

if fs: # if the file list is not empty
    
    # Get modified time, fail time and identify where the scrip has failed
    fsmod = {f: os.path.getmtime(fspath + '/' + f) for f in fs}
    failtime = max([v for k, v in fsmod.items()])
    failscrip = [k[:-4] for k, v in fsmod.items() if v == failtime][0]
    
    # now - porgram runtime
    floortime = (datetime.datetime.now() - datetime.timedelta(hours = hrs)).timestamp()
    
    if failtime < floortime:   # the pickles are old
#         [os.unlink(fn) for fn in all_pickles] # delete all the pickles
        [get_opt(ib, df) for df in dfs] # get options for all the symbols
    else:
        dfr = [df for df in dfs if dfs.symbol not in av_pkl_symbols] # pickle the remaining symbols

else:  # there are no pickles
    [get_opt(ib, df) for df in dfs] # get options for all the symbols