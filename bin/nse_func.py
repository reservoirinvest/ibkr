# p_nses.py
import pandas as pd
from ib_insync import *
from helper import catch

exchange = 'NSE'
fspath = '../data/nse/'

def p_nses(ib):
    '''Pickles nses underlying
    Arg: (ib) as connection object
    Returns: tuple of following dicts: 
        qualified underlying contracts nse underlying {symbol: contract}
        lots {symbol: lots}
        margins {symbol: margins}
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
    
    # !!!****DATA LIMITER***
    df_slm = df_slm[df_slm.ibSymbol.isin(['PNB', 'JETAIRWAY', 'BANKNIFTY', 'NIFTY50'])]
    #________________________

    # separate indexes and equities, eliminate discards from df_slm
    indexes = ['NIFTY50', 'BANKNIFTY']
    discards = ['NIFTYMID5', 'NIFTYIT', 'LUPIN']
    equities = sorted([s for s in df_slm.ibSymbol if s not in indexes+discards])
    
    # !!!****DATA LIMITER***
    equities = equities[0:2]

    symbols = equities+indexes

    cs = [Stock(s, exchange) if s in equities else Index(s, exchange) for s in symbols]

    qcs = ib.qualifyContracts(*cs) # qualified underlyings
    qcs = [q for c in qcs for q in ib.qualifyContracts(Contract(conId=c.conId))] # to get secType info
    qcs_dict = {q.symbol: q for q in qcs}

    lots_dict = [v for k, v in df_slm[['ibSymbol', 'lot']].set_index('ibSymbol').to_dict().items()][0]

    m_dict = [v for k, v in df_slm[['ibSymbol', 'margin']].set_index('ibSymbol').to_dict().items()][0] # from website

    # get the underlying prices
    tickers = ib.reqTickers(*qcs)
    undPrices = {t.contract.symbol: t.marketPrice() for t in tickers} # {symbol: undPrice}

    # get the option chains
    ch_list = [(q.symbol, 'IND', q.conId) 
               if q.symbol in indexes 
               else (q.symbol, 'STK', q.conId) 
               for q in qcs]

    chains = {s: ib.reqSecDefOptParams(underlyingSymbol=s, underlyingSecType=t, underlyingConId=c, futFopExchange='') for s, t, c in ch_list}

    # generate whatif contract-orders for margins
    co = [(k, min(v[0].expirations), 
      min(v[0].strikes, key=lambda 
          x:abs(x-undPrices[k])), 'P', 'NSE', lots_dict[k]) 
      for k, v in chains.items()]

    m_dict_ib = {c[0]:catch(lambda: ib.whatIfOrder(Option(c[0], c[1], c[2], c[3], c[4]), 
                 Order(action='SELL', orderType='MKT', totalQuantity=c[5], whatIf=True)).initMarginChange)
                 for c in co}

    margins = {k: max(v, float(m_dict_ib[k])) for k, v in m_dict.items() if k not in discards}
    
    # contracts, lots, margins, undPrices dataframe
    df_clmu = pd.concat([pd.DataFrame.from_dict(qcs_dict, orient='index'), 
               pd.DataFrame.from_dict(lots_dict, orient='index'), 
               pd.DataFrame.from_dict(margins, orient='index'),
               pd.DataFrame.from_dict(undPrices, orient='index')], axis=1, sort=False)

    df_clmu.columns=['contract', 'lot', 'margin', 'undPrice']
    df_clmu.to_pickle(fspath+'_lot_margin.pickle')
    
    return df_clmu

#_____________________________________

# p_nseopts.py
import numpy as np
import pandas as pd
from itertools import product, repeat

from helper import get_dte, get_maxfallrise, get_rollingmax_std

blk = 50
mindte = 3
maxdte = 60       # maximum days-to-expiry for options
minstdmult = 3    # minimum standard deviation multiple to screen strikes. 3 is 99.73% probability

def p_nseopts(ib, undContract, undPrice, lotsize, margin, fspath = '../data/nse/'):
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

    df_opt2 = df_opt2.assign(rom=df_opt2.optPrice*df_opt2.lotsize/df_opt2.optMargin*252/df_opt2.dte).sort_values('rom', ascending=False)
    
    # arrange the columns
    cols = ['optId', 'symbol', 'right', 'expiration', 'dte', 'strike', 'undPrice', 
            'lo52', 'hi52', 'Fall', 'Rise', 'loFall', 'hiRise', 'std3', 'loStd3', 'hiStd3', 
            'lotsize', 'optPrice', 'optMargin', 'rom']
    
    df_opt2 = df_opt2[cols]
    
    df_opt2.to_pickle(fspath+symbol+'.pkl')
    
    return None

#_____________________________________

# upd_nses.py
def upd_nses(ib, dfu):
    '''Updates the underlying nses
    Args:
       (ib) as connection object
       (dfu) as the underlying dataframe from p_nses
    Returns: None
       pickles back DataFrame with updated undPrice and margin'''
    
    # update prices
    tickers = ib.reqTickers(*dfu.contract)
    undPrices = {t.contract.symbol: t.marketPrice() for t in tickers} # {symbol: undPrice}

    # update margins - based on earliest expiration and strike closest to underlying price
    chains = {c.symbol: ib.reqSecDefOptParams(underlyingConId=c.conId, underlyingSecType=c.secType, underlyingSymbol=c.symbol, futFopExchange='')[0] for c in dfu.contract}

    lots_dict = dfu.lot.to_dict()

    # contract order for margin whatIf
    co = [(k, min(v.expirations), 
      min(v.strikes, key=lambda 
          x:abs(x-undPrices[k])), 'P', 'NSE', lots_dict[k]) 
      for k, v in chains.items()]

    # margin dictionary
    mdict = {c[0]: ib.whatIfOrder(
        Option(c[0], c[1], c[2], c[3], c[4]), 
        Order(action='SELL', orderType='MKT', totalQuantity=c[5], whatIf=True)
                          ).initMarginChange for c in co}

    # updates
    dfu['undPrice'].update(pd.Series(undPrices))
    dfu['margin'].update(pd.Series(mdict))

    # writeback
    dfu.to_pickle(fspath+'_lot_margin.pickle')

#_____________________________________

# remqty_nse.py
def remqty_nse(ib):
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
    c_dict = qlm[0]       # {symbol: contract}
    lots_dict = qlm[1]    # {symbol: lotsize}
    margin_dict = qlm[2]  # {symbol: margin}
    
    undContracts = [v for k, v in c_dict.items()]

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

