from ib_insync import util, Order
import pandas as pd
import numpy as np
import datetime
from math import sqrt, exp, log, erf, floor, log10, isnan
import logging

# logging to file
logpath = '../data/log/'

logger = logging.getLogger('__name__')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler(logpath+'helper.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# declarations
floor_dte = 30 # Minimum days expected for ilter_kxdte

# error catching for list comprehension
def catch(func, handle=lambda e : e, *args, **kwargs):
    '''List comprehension error catcher
    Args: 
        (func) as the function
         (handle) as the lambda of function
         <*args | *kwargs> as arguments to the functions
    Outputs:
        output of the function | <np.nan> on error
    Usage:
        eggs = [1,3,0,3,2]
        [catch(lambda: 1/egg) for egg in eggs]'''
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(handle(e))
        return np.nan

# gets days to expiry from now onwards
def get_dte(dt):
    '''Gets days to expiry
    Arg: (dt) as day in string format 'yyyymmdd'
    Returns: days to expiry as int'''
    return (util.parseIBDatetime(dt) - 
            datetime.datetime.now().date()).days

# get expected price percentage from DTEM
def get_hvstpricepct(expiry):
    '''Gets expected price percentage from DTE for harvesting trades.
    Assumes max DTE to be 30 days.
    Arg: (expiry) as string 'yyymmdd', e.g. from expPricePct 
    Returns: expected price percentage (xpp) as float
    Ref: http://interactiveds.com.au/software/Linest-poly.xls ... for getting curve function
    '''
    # if dte is to be extracted from contract.lastTradeDateOrContractMonth
    dte = get_dte(expiry)
    
    if dte > 30:
        dte = 30  # Forces the max DTE to be 30 days
    
    xpp = (103.6008 - 3.63457*dte + 0.03454677*dte*dte)/100
    
    return xpp

# get precision, based on the base
def get_prec(v, base):
    '''gives the precision value
    args:
       (v) as value needing precision in float
       (base) as the base value e.g. 0.05'''
    
    return round(round((v)/ base) * base, -int(floor(log10(base))))

# group options based on right, symbol and strike.
# this makes it easy to delete unwanted ones on inspection.
def grp_opts(df):
    '''Groups options and sorts strikes by puts and calls
    Arg: df as dataframe
    Returns: sorted dataframe'''
    
    gb = df.groupby('right')

    if 'C' in [k for k in gb.indices]:
        df_calls = gb.get_group('C').reset_index(drop=True).sort_values(['symbol', 'strike'], ascending=[True, True])
    else:
        df_calls =  pd.DataFrame([])

    if 'P' in [k for k in gb.indices]:
        df_puts = gb.get_group('P').reset_index(drop=True).sort_values(['symbol', 'strike'], ascending=[True, False])
    else:
        df_puts =  pd.DataFrame([])

    df = pd.concat([df_puts, df_calls]).reset_index(drop=True)
    
    return df

#... Black-Scholes
# Ref: - https://ideone.com/fork/XnikMm - Brian Hyde
#----------------------------------------------------
def get_bsm(undPrice, strike, dte, rate, volatility, divrate):
    ''' Gets Black Scholes output
    Args:
        (undPrice) : Current Stock Price in float
        (strike)   : Strike Price in float
        (dte)      : Days to expiration in float
        (rate)     : dte until expiry in days
        (volatility)    : Standard Deviation of stock's return in float
        (divrate)  : Dividend Rate in float
    Returns:
        (delta, call_price, put_price) as a tuple
    '''
    #statistics
    sigTsquared = sqrt(dte/365)*volatility
    edivT = exp((-divrate*dte)/365)
    ert = exp((-rate*dte)/365)
    d1 = (log(undPrice*edivT/strike)+(rate+.5*(volatility**2))*dte/365)/sigTsquared
    d2 = d1-sigTsquared
    Nd1 = (1+erf(d1/sqrt(2)))/2
    Nd2 = (1+erf(d2/sqrt(2)))/2
    iNd1 = (1+erf(-d1/sqrt(2)))/2
    iNd2 = (1+erf(-d2/sqrt(2)))/2

    #Outputs
    callPrice = round(undPrice*edivT*Nd1-strike*ert*Nd2, 2)
    putPrice = round(strike*ert*iNd2-undPrice*edivT*iNd1, 2)
    delta = Nd1

    return {'bsmCall': callPrice, 'bsmPut': putPrice, 'bsmDelta': delta}

#... max and min for calls and puts for a specific no of dte
def filter_kxdte(df, df_ohlc):
    '''Filters the strikes by dte*3, and, min of lows for Puts and max of highs for Calls
    Args: 
	   df as dataframe object
	   ohlc as ohlc dataframe object
       floor_dte as integer for minimum number of dtes to go backwards
    Returns: cleansed dfs without the risky strikes'''
    
    df['sfilt'] = [df_ohlc.set_index('symbol').loc[s][:max(floor_dte, d*3)].low.min() 
     if g == 'P' 
     else df_ohlc.set_index('symbol').loc[s][:max(floor_dte, d*3)].high.max() 
     for s, d, g in zip(df.symbol, df.dte, df.right)]
    
    df = df[((df.right == 'P') & (df.strike < df.sfilt)) | \
            ((df.right == 'C') & (df.strike > df.sfilt))]
    
    # return sorted df of puts and calls    
    df1 = grp_opts(df)
    
    # drop sfilt    
    return df1.drop('sfilt', axis=1)

#...filter options that are beyond 52 week highs and lows.
def get_hilo52(df):
    '''Keeps only options beyond 52 week high and low
    Arg: (df) as dataframe object
    Return: (df_opt) as dataframe with hilo52 options'''
    hilo_mask = ((df.right == 'P') & (df.strike < df.lo52)) | ((df.right == 'C') & (df.strike > df.hi52))
    df_opt = df[hilo_mask].reset_index(drop=True)
    return df_opt

#...keep puts only
def get_onlyputs(df):
    '''Keep only puts
    Arg: (df) as dataframe object
    Return: (df_opt) without calls'''
    
    return df[df.right=='P'].reset_index(drop=True)

#...keep puts only
def get_onlyputs(df):
    '''Keep only puts
    Arg: (df) as dataframe object
    Return: (df_opt) without calls'''
    
    return df[df.right=='P'].reset_index(drop=True)

##### Helper functions needing live ib
#______________________________________

# get the option margin for shorts
def get_opt_margin(ib, contract, mult=1):
    '''Gets margin of a single option contract
       Works for SNP only!
    Args:
        (contract) object as the option contract
        <mult> = 1 as int = no of contracts
    Returns: margin as a float'''
    
    order = Order(action='SELL', totalQuantity=mult, orderType='MKT')
    margin = float(ib.whatIfOrder(contract, order).initMarginChange)
    return margin
      
# function to get historical data of a contract
def get_hist(ib, contract, duration):
    
    '''Gets 1-day bars of contracts for the duration specified
    Args:
        (ib) as the active ib object
        (contract) as obj
        (duration) as int
    Returns: dataframe of symbol, date, ohlc, avg and volume 
    '''
    
    # Prepare the duration
    strduration = str(duration) + ' D'
    
    # Extract the history
    hist = ib.reqHistoricalData(contract=contract, endDateTime='', 
                                    durationStr=strduration, barSizeSetting='1 day',  
                                                whatToShow='Trades', useRTH=True)
    
    df = util.df(hist)
    df.insert(0, column='symbol', value=contract.symbol)
    
    return df

# function to get price and dividend ticker
def get_div_tick(ib, contract):
    '''Gets dividend ticker of the contract
    Arg: (contract) as a qualified contract object with conId
    Returns: ticker'''
    
    ib.reqMktData(contract, '456', snapshot=False, regulatorySnapshot=False) # request ticker stream

    ticker = ib.ticker(contract)
    
    # Ensure the ticker is filled
    while isnan(ticker.close):
        while ticker.dividends is None:
            ib.sleep(0.2)

    ib.cancelMktData(contract)
       
    return ticker
