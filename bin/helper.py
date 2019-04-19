# get_connected.py

from ib_insync import *
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

#_____________________________________

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

# get_snp_remqty.py
def get_snp_remqty(ib):
    '''generates the remaining quantities dictionary
    Args:
        (ib) as connection object
    Returns:
        remqty as a dictionary of {symbol: remqty}
        '''
    exchange = 'SMART'
    snp_assignment_limit = 50000
    lotsize=100
    
    # get the list of underlying contracts
    undContracts = get_snps(ib)
    c_dict = {u.symbol: u for u in undContracts} # {symbol: contract}
    
    tickers = ib.reqTickers(*undContracts)
    undPrices = {t.contract.symbol: t.marketPrice() for t in tickers} # {symbol: undPrice}

    df_und = \
        pd.DataFrame.from_dict(undPrices, orient='index', columns=['undPrice']).\
        join(pd.DataFrame.from_dict(c_dict, orient='index', columns=['undContract'])).dropna()
    
    df_und = df_und.assign(lotsize=lotsize)
    
    # remaining quantities in entire snp per assignment limit
    df_und = df_und.assign(remq=(snp_assignment_limit/(df_und.lotsize*df_und.undPrice)).astype('int')) 
    
    # from portfolio
    #_______________
    
    p = util.df(ib.portfolio()) # portfolio table
    
    # extract option contract info from portfolio table
    dfp = pd.concat([p, util.df([c for c in p.contract])[util.df([c for c in p.contract]).columns[:7]]], axis=1).iloc[:, 1:]
    dfp = dfp.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})

    # get the underlying's margins, lots, prices and positions
    pos_dict = dfp.groupby('symbol')['position'].sum().to_dict()
    p_syms = {s for s in dfp.symbol}
    p_undc = {s: ib.qualifyContracts(Stock(s, exchange, currency='USD')) for s in p_syms}   # {symbol: contracts} dictionary
    undmlist = [get_margins(ib, u, 100) for u in p_undc.values()]                 # {contract: margin} dictionary
    p_undMargins = {k.symbol: v for i in undmlist for k, v in i.items()}        # {symbol: margin} dictionary
    p_undPrices = {u: undPrices[u] for u in p_syms}    #{symbol: undPrice} dictionary
    
    dfp1 = pd.DataFrame.from_dict(p_undc, orient='index', columns=['contract']). \
        join(pd.DataFrame.from_dict(p_undMargins, orient='index', columns=['undmargin'])). \
        join(pd.DataFrame.from_dict(p_undPrices, orient='index', columns=['undPrice'])). \
        join(pd.DataFrame.from_dict(pos_dict, orient='index', columns=['position']))
    
    dfp1 = dfp1.assign(undLot=1)
    
    dfp1 = dfp1.assign(qty=(dfp1.position/dfp1.undLot).astype('int'))
    
    # make the blacklist
    #___________________
    remqty_dict = pd.DataFrame(df_und.loc[dfp1.index].remq + dfp1.qty).to_dict()[0] # remq from portfolio
    remqty_dict = {k:(v if v > 0 else 0) for k, v in remqty_dict.items()} # portfolio's remq with negative values removed
    blacklist = [k for k, v in remqty_dict.items() if v <=0] # the blacklist
    df_und.remq.update(pd.Series(remqty_dict)) # replace all underlying with remq of portfolio
    remqty = df_und.remq.to_dict() # dictionary
    return remqty

#_____________________________________

# get_snp_options.py
from itertools import product, repeat

blk = 50
mindte = 3
maxdte = 60       # maximum days-to-expiry for options
minstdmult = 3    # minimum standard deviation multiple to screen strikes. 3 is 99.73% probability

def get_snp_options(ib, undContract, undPrice, fspath = '../data/snp/'):
    '''Pickles the option chains
    Args:
        (ib) ib connection as object
        (undContract) underlying contract as object
        (undPrice) underlying contract price as float'''
    
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

# get_prec.py
# get precision, based on the base
from math import floor, log10

def get_prec(v, base):
    '''gives the precision value
    args:
       (v) as value needing precision in float
       (base) as the base value e.g. 0.05'''
    
    return round(round((v)/ base) * base, -int(floor(log10(base))))

#_____________________________________

# get_nses.py

import pandas as pd

blk = 50 # no of stocks in a block
exchange = 'NSE'
def get_nses(ib):
    '''Returns: list of nse underlying (contracts, dictionary of lots, dictionary of margins) as a tuple
    '''
    tp = pd.read_html('https://www.tradeplusonline.com/Equity-Futures-Margin-Calculator.aspx')

    df_tp = tp[1][2:].iloc[:, :-1]
    df_tp = df_tp.iloc[:, [0,1,5]]
    df_tp.columns=['nseSymbol', 'lot', 'margin']

    cols = df_tp.columns.drop('nseSymbol')
    df_tp[cols] = df_tp[cols].apply(pd.to_numeric, errors='coerce') # convert lot and margin to numeric

    df_slm = df_tp.copy()

    # Truncate to 9 characters for ibSymbol
    df_slm['ibSymbol'] = df_slm.nseSymbol.str.slice(0,9)

    # nseSymbol to ibSymbol dictionary for conversion
    ntoi = {'M&M': 'MM', 'M&MFIN': 'MMFIN', 'L&TFH': 'LTFH', 'NIFTY': 'NIFTY50'}

    # remap ibSymbol, based on the dictionary
    df_slm.ibSymbol = df_slm.ibSymbol.replace(ntoi)

    # separate indexes and equities, eliminate discards from df_slm
    indexes = ['NIFTY50', 'BANKNIFTY']
    discards = ['NIFTYMID5', 'NIFTYIT', 'LUPIN']
    equities = sorted([s for s in df_slm.ibSymbol if s not in indexes+discards])

    symbols = equities+indexes

    cs = [Stock(s, exchange) if s in equities else Index(s, exchange) for s in symbols]

    qcs = ib.qualifyContracts(*cs) # qualified underlyings

    lots_dict = [v for k, v in df_slm[['ibSymbol', 'lot']].set_index('ibSymbol').to_dict().items()][0]
    
    margins_dict = [v for k, v in df_slm[['ibSymbol', 'margin']].set_index('ibSymbol').to_dict().items()][0]
    
    return qcs, lots_dict, margins_dict

#_____________________________________

# get_nse_options.py
from itertools import product, repeat

blk = 50
mindte = 3
maxdte = 60       # maximum days-to-expiry for options
minstdmult = 3    # minimum standard deviation multiple to screen strikes. 3 is 99.73% probability

def get_nse_options(ib, undContract, undPrice, lotsize, margin, fspath = '../data/nse/'):
    '''Pickles the option chains
    Args:
        (ib) ib connection as object
        (undContract) underlying contract as object
        (undPrice) underlying contract price as float
        (lotsize) lot-size as float
        (margin) margin of undContract (used as a surrogate for option)
        <fspath> file path to store nse options'''
    
    symbol = undContract.symbol
    
    chains = ib.reqSecDefOptParams(underlyingSymbol = symbol,
                         futFopExchange = '',
                         underlyingSecType = undContract.secType,
                         underlyingConId= undContract.conId)

    xs = [set(product(c.expirations, c.strikes)) for c in chains if c.exchange == 'NSE']

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

    optipl = [Option(s, e, k, r, 'NSE') for s, e, k, r in zip(df_opt1.symbol, df_opt1.expiration, df_opt1.strike, df_opt1.right)]

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

    # Get lotsize and margin for the underlying symbol
    df_opt2 = df_opt2.assign(lotsize = lotsize)
    df_opt2 = df_opt2.assign(optMargin = margin)

    opt_contracts = [opt_iDict[i] for i in df_opt2.optId]

#     opt_margins = get_margins(ib,opt_contracts, lotsize)
#     df_opt2 = df_opt2.assign(optMargin = [abs(v) for k, v in opt_margins.items()])

    df_opt2 = df_opt2.assign(rom=df_opt2.optPrice*df_opt2.lotsize/df_opt2.optMargin*252/df_opt2.dte).sort_values('rom', ascending=False)

    df_opt2.to_pickle(fspath+symbol+'.pkl')

#_____________________________________

# get_nse_remqty.py
def get_nse_remqty(ib):
    '''generates the remaining quantities dictionary
    Args:
        (ib) as connection object
    Returns:
        remqty as a dictionary of {symbol: remqty}
        '''
    exchange = 'NSE'
    nse_assignment_limit = 1500000

    # get the list of underlying contracts and dictionary of lots
    qlm = get_nses(ib)
    undContracts = qlm[0]
    c_dict = {u.symbol: u for u in undContracts} # {symbol: contract}
    lots_dict = qlm[1]                # {symbol: lotsize}
    margin_dict = qlm[2]              # {symbol: margin}

    tickers = ib.reqTickers(*undContracts)
    undPrices = {t.contract.symbol: t.marketPrice() for t in tickers} # {symbol: undPrice}
    
    df_und = \
        pd.DataFrame.from_dict(lots_dict, orient='index', columns=['lotsize']).\
        join(pd.DataFrame.from_dict(undPrices, orient='index', columns=['undPrice'])).\
        join(pd.DataFrame.from_dict(c_dict, orient='index', columns=['undContract'])).dropna()
    
    df_und = df_und.assign(remq=(nse_assignment_limit/(df_und.lotsize*df_und.undPrice)).astype('int')) # remaining quantities in entire nse
    
    # from portfolio
    #_______________
    
    p = util.df(ib.portfolio()) # portfolio table
    
    # extract option contract info from portfolio table
    dfp = pd.concat([p, util.df([c for c in p.contract])[util.df([c for c in p.contract]).columns[:7]]], axis=1).iloc[:, 1:]
    dfp = dfp.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})

    # get the underlying's margins, lots, prices and positions
    pos_dict = dfp.groupby('symbol')['position'].sum().to_dict()
    p_syms = {s for s in dfp.symbol}
    p_undc = {s: ib.qualifyContracts(Stock(s, exchange)) for s in p_syms}   # {symbol: contracts} dictionary
    p_undMargins = {u: margin_dict[u] for u in p_syms} # {symbol: undMargin} dictionary
    p_undLots = {u: lots_dict[u] for u in p_syms}      # {symbol: lots} dictionary
    p_undPrices = {u: undPrices[u] for u in p_syms}    #{symbol: undPrice} dictionary
    
    dfp1 = pd.DataFrame.from_dict(p_undc, orient='index', columns=['contract']). \
        join(pd.DataFrame.from_dict(p_undMargins, orient='index', columns=['undmargin'])). \
        join(pd.DataFrame.from_dict(p_undLots, orient = 'index', columns=['undLot'])). \
        join(pd.DataFrame.from_dict(p_undPrices, orient='index', columns=['undPrice'])). \
        join(pd.DataFrame.from_dict(pos_dict, orient='index', columns=['position']))
    
    dfp1 = dfp1.assign(qty=(dfp1.position/dfp1.undLot).astype('int'))
    
    # make the blacklist
    #___________________
    remqty_dict = pd.DataFrame(df_und.loc[dfp1.index].remq + dfp1.qty).to_dict()[0] # remq from portfolio
    remqty_dict = {k:(v if v > 0 else 0) for k, v in remqty_dict.items()} # portfolio's remq with negative values removed
    blacklist = [k for k, v in remqty_dict.items() if v <=0] # the blacklist
    df_und.remq.update(pd.Series(remqty_dict)) # replace all underlying with remq of portfolio
    remqty = df_und.remq.to_dict() # dictionary
    return remqty

#_____________________________________

# recalc_opts.py
def recalc_opts(ib, df):
    '''Recalculates option prices and margins
    Arg:
       (ib) as connection object
       (df) as the dataframe of options
    Returns: dataframe with updated OptPrices and OptMargins
    '''
    opt_contracts = [Contract(conId=c) for c in df.optId]
    qopts = ib.qualifyContracts(*opt_contracts)
    tickers = ib.reqTickers(*qopts)
    optPrices = {t.contract.conId: t.marketPrice() for t in tickers} # {symbol: undPrice}
    optMargins = {c.conId: ib.whatIfOrder(c,Order(action='SELL', orderType='MKT', totalQuantity=abs(ls), whatIf=True)).initMarginChange 
        for c, ls in (zip(qopts, df.lotsize))}
    
    # update prices and margins
    df = df.set_index('optId')
    df.optPrice.update(pd.Series(optPrices))
    df.optMargin.update(pd.Series(optMargins))
    df.optMargin = df.optMargin.astype('float')
    df = df.assign(rom=df.optPrice*df.lotsize/df.optMargin*252/df.dte)
    
    return df.reset_index()

#_____________________________________

