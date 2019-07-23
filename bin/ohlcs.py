# ohlcs.py
"""Program that generates ohlcs (both NSE and SNP)
Date: 23-June-2019
Ver: 1.0
Time taken: 12 mins
"""
from z_helper import *  # needed for util

def ohlcs(ib, id_sym, fspath, logpath):
    '''Get ohlcs with stDev (8 mins)
    Args:
        (ib) as connection object
        (id_sym) as {undId: 'symbol'} dictionary
        (logpath) as string for path for lots
        (fspath) as string for path of pickles
    Returns:
        ohlcs dataframe with stDev'''

    with open(logpath+'ohlc.log', 'w'):
        pass # clear the run log

    util.logToFile(logpath+'ohlc.log')

    ohlcs = []
    with tqdm(total= len(id_sym), file=sys.stdout, unit= 'symbol') as tqh:
        for k, v in id_sym.items():
            tqh.set_description(f"Getting OHLC hist frm IBKR for {v.ljust(9)}")
            ohlcs.append(catch(lambda:do_hist(ib, k, fspath)))
            tqh.update(1)

    # Remove nan from ohlcs list
    li = [o for o in ohlcs if str(o) != 'nan']
    
    df_ohlcs = pd.concat(li).reset_index(drop=True)
    
    return df_ohlcs

#_____________________________________

