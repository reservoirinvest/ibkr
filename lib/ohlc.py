import pandas as pd

from helper import get_hist
from math import sqrt

tradingdays = 252

def get_ohlc(ib, contract, fspath):
    '''Gets the ohlc
    Args: 
       (ib) as ib object
       (contract) as contract object
       (fspath) as string containing zdata path
    Returns:
       None. But pickles ohlc'''
    
    df_ohlc = get_hist(ib, contract, 365).set_index('date').sort_index(ascending = False)
    
    # get cumulative standard deviation
    df_stdev = pd.DataFrame(df_ohlc['close'].expanding(1).std(ddof=0))
    df_stdev.columns = ['stdev']

    # get cumulative volatility
    df_vol = pd.DataFrame(df_ohlc['close'].pct_change().expanding(1).std(ddof=0)*sqrt(tradingdays))
    df_vol.columns = ['volatility']

    df_ohlc1 = df_ohlc.join(df_vol)

    df_ohlc2 = df_ohlc1.join(df_stdev)

    #pickle the ohlc
    df_ohlc2.to_pickle(fspath+contract.symbol+'_ohlc.pkl')
    
    return df_ohlc2