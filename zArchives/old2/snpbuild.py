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
import time
import pandas as pd

start = time.time()

# declaration
hrs = 3   # approximate time taken to get options for all scrips

# qualified list of stocks and index, with their contracts and chains
df_snplist = snp_list(ib)
# df_snplist = pd.read_pickle(fspath+'snplist.pkl')

# make the dfs from snplist
dfs = [df_snplist[df_snplist.symbol == s].reset_index(drop=True) for s in list(df_snplist.symbol.unique())]

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
        dfr = [df for df in dfs for sym in df.symbol.unique() if sym not in av_pkl_symbols] # pickle the remaining symbols

        [get_opt(ib, df) for df in dfr] # get options for all the symbols

else:  # there are no pickles
    [get_opt(ib, df) for df in dfs] # get options for all the symbols

end = time.time()

time_taken = end-start

m, s = divmod(time_taken, 60)
h, m = divmod(m, 60)

print("SNP build completed in {0:2.0f} hours : {1:2.0f} mins : {2:2.0f} seconds!".format(h, m, s))
ib.disconnect()