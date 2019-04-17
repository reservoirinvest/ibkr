# get_margins.py
from itertools import repeat
def get_margins(ib, contracts, *lotsize):
    '''Margin dictionary. 1 min for 100 contracts.
    Args:
        (ib) as object
        (contracts) as <series>|<list> of underlying contracts
        (*lotsize) as <int>|<list>
    Returns:
        {contract (obj): underlying_margin(float)} as dictionary'''
    
    if type(contracts) is pd.Series:
        contracts = list(contracts)
    else:
        contracts = contracts

    if type(lotsize[0]) is pd.Series:
        positions = list(lotsize[0])
    else:
        positions = repeat(lotsize[0], len(contracts)) # convert *arg tuple to int
    
    orders = [Order(action='SELL', orderType='MKT', totalQuantity=abs(p), whatIf=True)
              if p < 0 else
              Order(action='BUY', orderType='MKT', totalQuantity=abs(p), whatIf=True)
              for p in positions]

    co = [c for c in zip(contracts, orders)]

    dict_margins = {c: float(ib.whatIfOrder(c, o).initMarginChange) for c, o in co}
    
    return dict_margins

#_____________________________________

# get_snps.py
import pandas as pd
blk = 50 # no of stocks in a block
def get_snps(ib):
    '''Returns: list of underlying contracts
    Usage: 
       with get_connected('snp', 'live') as ib: und_contracts = get_snps(ib)'''

    # exclusion list
    excl = ['VXX','P', 'TSRO']

    # Download cboe weeklies to a dataframe
    dls = "http://www.cboe.com/publish/weelkysmf/weeklysmf.xls"

#     snp500 = list(pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0][1:].loc[:, 1])

    snp100 = pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')[1][1:].loc[:, 0]
    snp100 = [s.replace('.', ' ') if '.' in s else s  for s in snp100] # without dot in symbol
    # read from row no 11, dropna and reset index
    df_cboe = pd.read_excel(dls, header=12, 
                            usecols=[0,2,3]).loc[11:, :]\
                            .dropna(axis=0)\
                            .reset_index(drop=True)

    # remove column names white-spaces and remap to IBKR
    df_cboe.columns = df_cboe.columns.str.replace(' ', '')

    # remove '/' for IBKR
    df_cboe.Ticker = df_cboe.Ticker.str.replace('/', ' ', regex=False)

    # make symbols
    symbols = {s for s in df_cboe.Ticker if s not in excl if s in snp100}
    stocks = [Stock(symbol=s, exchange='SMART', currency='USD') for s in symbols]

    stkblks = [stocks[i: i+blk] for i in range(0, len(stocks), blk)] # blocks of stocks

    # qualify the contracts
    contracts = [ib.qualifyContracts(*s) for s in stkblks]

    # return flattened contract list
    return [contract for subl in contracts for contract in subl] 

#_____________________________________

# get_dte.py
import datetime

def get_dte(dt):
    '''Gets days to expiry
    Arg: (dt) as day in string format 'yyyymmdd'
    Returns: days to expiry as int'''
    return (util.parseIBDatetime(dt) - 
            datetime.datetime.now().date()).days

#_____________________________________

# get_rollingmax_std.py
from math import sqrt
tradingdays = 252

def get_rollingmax_std(ib, c, dte, durmult=3):
    '''gets the rolling max standard deviation
    Args:
        (ib) as connection object
        (c) as contract object
        (dte) as int for no of days for expiry
        (durmult) no of samples to go backwards on
    Returns:
        maximum rolling standard deviation as int'''

    durStr = str(durmult*dte) + ' D' # Duration String
    
    # Extract the history
    hist = ib.reqHistoricalData(contract=c, endDateTime='', 
                                    durationStr=durStr, barSizeSetting='1 day',  
                                                whatToShow='Trades', useRTH=True)
    df = util.df(hist)
    df.insert(0, column='symbol', value=c.symbol)

    df_ohlc = df.set_index('date').sort_index(ascending = False)

    # get cumulative standard deviation
    df_stdev = pd.DataFrame(df_ohlc['close'].expanding(1).std(ddof=0))
    df_stdev.columns = ['stdev']

    # get cumulative volatility
    df_vol = pd.DataFrame(df_ohlc['close'].pct_change().expanding(1).std(ddof=0)*sqrt(tradingdays))
    df_vol.columns = ['volatility']

    df_ohlc1 = df_ohlc.join(df_vol)

    df_ohlc2 = df_ohlc1.join(df_stdev)

    return df_stdev.stdev.max()

#_____________________________________

# get_maxfallrise.py
def get_maxfallrise(ib, c, dte):
    '''get the maximum rise, fall for rolling window of dte and lo52, hi52
    Args:
       (ib) as connection object
       (c) as the underlying contract object
       (dte) as int for days to expiry of a contract
    Returns:
       (lo52, hi52, max_fall, max_rise) tuple of floats'''
    
    
    hist = ib.reqHistoricalData(contract=c, endDateTime='', 
                                        durationStr='365 D', barSizeSetting='1 day',  
                                                    whatToShow='Trades', useRTH=True)

    df = util.df(hist)
    df.insert(0, column='symbol', value=c.symbol)

    df_ohlc = df.set_index('date').sort_index(ascending=True)
    df = df_ohlc.assign(delta=df_ohlc.high.rolling(dte).max()-df_ohlc.low.rolling(dte).min(), pctchange=df_ohlc.high.pct_change(periods=dte))

    df1 = df.sort_index(ascending=False)
    max_fall = df1[df1.pctchange<=0].delta.max()
    max_rise = df1[df1.pctchange>0].delta.max()
    hi52 = df1.high.max()
    lo52 = df1.low.min()
    
    return(lo52, hi52, max_fall, max_rise)

#_____________________________________

# catch.py
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
        return np.nan

#_____________________________________

# save_open_orders.py
def save_open_orders(ib, fspath='../data/snp/'):
    '''Saves the open orders, before deleting from the system
    Arg: 
        (ib) as connection object
        (fspath) to save the files
    Returns: None'''
    import pandas as pd
    import glob

#     fspath = '../data/snp/'
    fn = '_openorders'
    fe = '.pkl'

    filepresent = glob.glob(fspath+fn+fe)

    if filepresent:
        qn = input(f'File {[f for f in filepresent]} is available. Overwrite?: Y/N ')
        if qn.upper() != 'Y':
            return

    ib.reqAllOpenOrders()
    opTrades = ib.openTrades()

    if opTrades:
        dfopen = pd.DataFrame.from_dict({o.contract.conId: \
        (o.contract.symbol, \
        o.contract.lastTradeDateOrContractMonth, \
        o.contract.right, \
        o.order.action, \
        o.order.totalQuantity, \
        o.order.orderType, \
        o.order.lmtPrice, \
        o.contract, \
        o.order) \
        for o in opTrades}).T.reset_index()

        dfopen.columns = ['conId', 'symbol', 'expiration', 'right', 'action', 'qty', 'orderType', 'lmtPrice', 'contract', 'order']
        dfopen.to_pickle(fspath+fn+fe)
        return dfopen
    else:
        return None

#_____________________________________

# get_portfolio.py
import numpy as np
import pandas as pd

def get_portfolio(ib):
    '''Arg: (ib) as the connection object
       Returns: (df) as the dataframe with margins and assignments'''

    p = util.df(ib.portfolio()) # portfolio table

    # extract option contract info from portfolio table
    dfp = pd.concat([p, util.df([c for c in p.contract])[util.df([c for c in p.contract]).columns[:7]]], axis=1).iloc[:, 1:]
    dfp = dfp.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})

    # get the underlying's margins
    syms = {s for s in dfp.symbol}
    undc = {s: ib.qualifyContracts(Stock(s, 'SMART', 'USD')) for s in syms}   # {symbol: contracts} dictionary

    undmlist = [get_margins(ib, u, 1) for u in undc.values()]                 # {contract: margin} dictionary
    undMargins = {k.symbol: v for i in undmlist for k, v in i.items()}        # {symbol: margin} dictionary

    dfp = dfp.assign(lotsize=np.where(dfp.secType == 'OPT', 100, 1))

    dfp = dfp.assign(undMargin = dfp.symbol.map(undMargins)*dfp.position*dfp.lotsize)

    #...get the underlying prices
    undContracts = [j for k, v in undc.items() for j in v]
    tickers = ib.reqTickers(*undContracts)
    undPrices = {t.contract.symbol: t.marketPrice() for t in tickers}
    dfp = dfp.assign(undPrice = dfp.symbol.map(undPrices))


    # get the contracts again (as some of them miss markets)
    port_c = [j for c in [ib.qualifyContracts(Contract(conId=c)) for c in dfp.conId] for j in c]

    dfp = dfp.assign(contract=port_c)

    # get the portfolio margins
    dict_port_opt_margins = get_margins(ib, dfp.contract, dfp.position)

    dfp = dfp.assign(margin=dfp.contract.map(dict_port_opt_margins)*dfp.position/abs(dfp.position))  # margin's sign put in line with position held

    dfp = dfp.assign(assignment = dfp.undPrice*dfp.position*dfp.lotsize).drop(['account', 'multiplier'], axis=1)
    
    return dfp

#_____________________________________

# get_p_remqty.py

from ib_insync import *

assignment_limit = 120000

def get_p_remqty(ib):
    '''gets remaining quantity for target options
    This has to be run before get_remqty
    Args: 
        (ib) as connection object
    Returns: (remqty) as dictionary {symbol: value}
    Dependencies: save_open_orders(), get_portfolio(), get_snps()'''
    
    #... read the account info
    ac = ib.accountValues()
    df_a = util.df(ac)

    #... set max margin per position
    net_liq = float(df_a[df_a.tag == 'NetLiquidation'].iloc[0].value) 
    av_funds = float(df_a[df_a.tag == 'FullAvailableFunds'].iloc[0].value)

    # save any openorders into a pickle
    if ib.reqAllOpenOrders():
        dfopenords = save_open_orders(ib)

    # cancel all openorders. This is to prevent get_margins() from failing
    cancelTrades = ib.reqGlobalCancel()

    dfp = get_portfolio(ib)  # get the portfolio positions

    current_assignment_dict = dfp.groupby('symbol').sum()[['assignment']].to_dict()['assignment'] # current assignment possibility

    df_ua = dfp.drop_duplicates('symbol')[['symbol', 'undPrice', 'lotsize']] # unit assignment
    df_ua = df_ua.assign(assignment=df_ua.undPrice*df_ua.lotsize)
    unit_assignment_dict = df_ua.groupby('symbol').mean()[['assignment']].to_dict()['assignment']

    remqty_dict = {k1: (assignment_limit+v1)/v2 
                   for k1, v1 in current_assignment_dict.items() 
                   for k2, v2 in unit_assignment_dict.items() if k1 == k2}

    rqwoz = [(k, int(v)) if v > 0 else (k, 0) 
             for k, v in remqty_dict.items()] # remaining quantity without zeros
    remqty_p = {k: v for (k, v) in rqwoz} # remaining quantity from positions
    
    return remqty_p

#_____________________________________

# get_snp_options.py
from itertools import product, repeat

blk = 50
mindte = 3
maxdte = 60       # maximum days-to-expiry for options
minstdmult = 3    # minimum standard deviation multiple to screen strikes. 3 is 99.73% probability
fspath = '../data/snp/' # path for pickles

def get_snp_options(ib, undContract, undPrice, fspath = '../data/snp/'):
    '''Pickles the option chains
    Args:
        (ib) ib connection as object
        (undContract) underlying contract as object
        (undPrice) underlying contract price as float'''
    
#     fspath = '../data/snp/' # path for pickles
    
    symbol = undContract.symbol
    
    chains = ib.reqSecDefOptParams(underlyingSymbol = symbol,
                         futFopExchange = '',
                         underlyingSecType = undContract.secType,
                         underlyingConId= undContract.conId)

    xs = [set(product(c.expirations, c.strikes)) for c in chains if c.exchange == 'SMART']

    expirations = [i[0] for j in xs for i in j]
    strikes = [i[1] for j in xs for i in j]
    dflength = len(expirations)

    #...first df with symbol, strike and expiry
    df1 = pd.DataFrame({'cid': pd.Series(np.repeat(undContract.conId,dflength)), 
                  'symbol': pd.Series(np.repeat(symbol,dflength)),
                  'expiration': expirations,
                  'strike': strikes,
                  'dte': [get_dte(e) for e in expirations],
                  'undPrice': pd.Series(np.repeat(undPrice,dflength))})

    df2 = df1[(df1.dte > mindte) & (df1.dte < maxdte)].reset_index(drop=True)  # limiting dtes

    dtes = df2.dte.unique().tolist()

    #...get the max fall / rise for puts / calls
    maxFallRise = {d: get_maxfallrise(ib, c, d) for c, d in zip(repeat(undContract), dtes)}

    df3 = df2.join(pd.DataFrame(df2.dte.map(maxFallRise).tolist(), index=df2.index, columns=['lo52', 'hi52', 'Fall', 'Rise']))

    df4 = df3.assign(loFall = df3.undPrice-df3.Fall, hiRise = df3.undPrice+df3.Rise)

    std = {d: get_rollingmax_std(ib, c, d) for c, d in zip(repeat(undContract), dtes)}

    df4['std3'] = df4.dte.map(std)*minstdmult

    df4['loStd3'] = df4.undPrice - df4.std3
    df4['hiStd3'] = df4.undPrice + df4.std3

    # flter puts and calls by standard deviation
    df_puts = df4[df4.strike < df4.loStd3]
    df_calls = df4[df4.strike > df4.hiStd3]

    # df_puts = df4 # keep the puts dataframe without limits
    # df_calls = df4.iloc[0:0] # empty the calls dataframe

    # with rights
    df_puts = df_puts.assign(right='P')
    df_calls = df_calls.assign(right='C')

    # qualify the options
    df_opt1 = pd.concat([df_puts, df_calls]).reset_index()

    optipl = [Option(s, e, k, r, 'SMART') for s, e, k, r in zip(df_opt1.symbol, df_opt1.expiration, df_opt1.strike, df_opt1.right)]

    optblks = [optipl[i: i+blk] for i in range(0, len(optipl), blk)] # blocks of optipl

    # qualify the contracts
    contracts = [ib.qualifyContracts(*s) for s in optblks]
    q_opt = [d for c in contracts for d in c]

    opt_iDict = {c.conId: c for c in q_opt}

    df_opt1 = util.df(q_opt).loc[:, ['conId', 'symbol', 'lastTradeDateOrContractMonth', 'strike', 'right']]

    df_opt1 = df_opt1.rename(columns={'lastTradeDateOrContractMonth': 'expiration', 'conId': 'optId'})

    opt_tickers = ib.reqTickers(*q_opt)
#     ib.sleep(1) # to get the tickers filled

    df_opt1 = df_opt1.assign(optPrice = [t.marketPrice() for t in opt_tickers])

    df_opt1 = df_opt1[df_opt1.optPrice > 0.0]

    cols=['symbol', 'expiration', 'strike']
    df_opt2 = pd.merge(df4, df_opt1, on=cols).drop('cid', 1).reset_index(drop=True)

    # Make lotsize -ve for puts and +ve for calls for margin calculation
    df_opt2 = df_opt2.assign(lotsize = 1)

    opt_contracts = [opt_iDict[i] for i in df_opt2.optId]

    lotsize = pd.Series([l for l in df_opt2.lotsize])

    opt_margins = get_margins(ib,opt_contracts, lotsize)

    df_opt2 = df_opt2.assign(optMargin = [abs(v) for k, v in opt_margins.items()])

    df_opt2 = df_opt2.assign(rom=df_opt2.optPrice/df_opt2.optMargin*252/df_opt2.dte).sort_values('rom', ascending=False)

    df_opt2.to_pickle(fspath+symbol+'.pkl')

#_____________________________________

# get_snp_remqty.py
def get_snp_remqty(ib, remqty_p, undContracts):
    '''generates the remaining quantities dictionary
    Args:
        (ib) as connection object
        (remqty_p) remaining quantity from portfolio as dictionary
        (undContracts) underlying contracts as a list
    Returns:
        remqty as a dictionary of {symbol: remqty}
        '''
    lotsize=100
    
    tickers = ib.reqTickers(*undContracts)
    undPrices = {t.contract.symbol: t.marketPrice() for t in tickers} # {symbol: undPrice}

    remq = {k: max(1, int(assignment_limit/v/lotsize)) for k, v in undPrices.items()} # maximum given to give GOOG, BKNG, etc a chance!
    remq_list = [(k, remqty_p[k]) if k in remqty_p.keys() else (k, v) for k, v in remq.items()]
    remqty = {k: v for (k, v) in remq_list}
    return remqty

#_____________________________________

# grp_opts.py
# group options based on right, symbol and strike.
# this makes it easy to delete unwanted ones on inspection.
import pandas as pd

def grp_opts(df):
    '''Groups options and sorts strikes by puts and calls
    Arg: 
       df as dataframe. Requires 'symbol', 'strike' and 'dte' fields in the df
    Returns: sorted dataframe'''
    
    gb = df.groupby('right')

    if 'C' in [k for k in gb.indices]:
        df_calls = gb.get_group('C').reset_index(drop=True).sort_values(['symbol', 'dte', 'strike'], ascending=[True, False, True])
    else:
        df_calls =  pd.DataFrame([])

    if 'P' in [k for k in gb.indices]:
        df_puts = gb.get_group('P').reset_index(drop=True).sort_values(['symbol', 'dte', 'strike'], ascending=[True, False, False])
    else:
        df_puts =  pd.DataFrame([])

    df = pd.concat([df_puts, df_calls]).reset_index(drop=True)
    
    return df

#_____________________________________

