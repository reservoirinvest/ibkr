# p_snps.py
import pandas as pd
from snp_variables import blk
from ib_insync import *

def p_snps(ib):
    '''Pickles snps underlying
    Arg: (ib) as connection object
    Returns: Dataframe of symbol, lot, margin with underlying info'''

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
    contracts = [contract for subl in contracts for contract in subl]
    qcs_dict = {q.symbol: q for q in contracts}

    # get the margins
    orders = [Order(action='SELL', orderType='MKT', totalQuantity=1, whatIf=True)]*len(contracts)
    margins = [ib.whatIfOrder(c, o).initMarginChange for c, o in zip(contracts, orders)]
    m_dict = {s.symbol:m for s, m in zip(contracts, margins)}

    # get the undPrices
    tickers = ib.reqTickers(*contracts)
    undPrices = {t.contract.symbol: t.marketPrice() for t in tickers} # {symbol: undPrice}

    qcs_dict = {q.symbol: q for q in contracts}

    # contracts, lots, margins, undPrices dataframe
    df_clmu = pd.DataFrame.from_dict(qcs_dict, orient='index', columns=['contract']).\
             join(pd.DataFrame.from_dict(m_dict, orient='index', columns=['margin'])).\
             join(pd.DataFrame.from_dict(undPrices, orient='index', columns=['undPrice']))

    df_clmu = df_clmu.assign(lot=100)

    df_clmu = df_clmu.assign(margin=abs(pd.to_numeric(df_clmu.margin)).astype('int'))
    
    df_clmu.columns=['contract', 'lot', 'margin', 'undPrice']
    df_clmu.to_pickle(fspath+'_lot_margin.pickle')
    
    return df_clmu

#_____________________________________

# p_snpopts.py
import numpy as np
import pandas as pd
from itertools import product, repeat

from helper import get_dte, get_maxfallrise, get_rollingmax_std
from snp_variables import *

def p_snpopts(ib, undContract, undPrice, margin, lotsize=100):
    '''Pickles the option chains
    Args:
        (ib) ib connection as object
        (undContract) underlying contract as object
        (undPrice) underlying contract price as float
        (lotsize) lot-size as float is 100
        (margin) margin of undContract (used as a surrogate for option)
        <fspath> file path to store snp options'''
    
    symbol = undContract.symbol
    
    chains = ib.reqSecDefOptParams(underlyingSymbol = symbol,
                         futFopExchange = '',
                         underlyingSecType = undContract.secType,
                         underlyingConId= undContract.conId)

    xs = [set(product(c.expirations, c.strikes)) for c in chains if c.exchange == exchange]

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

    optipl = [Option(s, e, k, r, exchange) for s, e, k, r in zip(df_opt1.symbol, df_opt1.expiration, df_opt1.strike, df_opt1.right)]

    optblks = [optipl[i: i+blk] for i in range(0, len(optipl), blk)] # blocks of optipl

    # qualify the contracts
    contracts = [ib.qualifyContracts(*s) for s in optblks]
    
    try:
        [z for x in contracts for y in x for z in y] # If an Option is available, this will fail
        if not [z for x in contracts for y in x for z in y]:  # Check if there is anything in the list
            return None  # The list is empty in 3 levels!
    except TypeError as e:
        pass

    q_opt = [d for c in contracts for d in c]

    opt_iDict = {c.conId: c for c in q_opt}

    df_opt1 = util.df(q_opt).loc[:, ['conId', 'symbol', 'lastTradeDateOrContractMonth', 'strike', 'right']]

    df_opt1 = df_opt1.rename(columns={'lastTradeDateOrContractMonth': 'expiration', 'conId': 'optId'})

    opt_tickers = ib.reqTickers(*q_opt)

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

# upd_snps.py
def upd_snps(ib, dfu):
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
          x:abs(x-undPrices[k])), 'P', 'SMART', lots_dict[k]) 
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
    
    return dfu

#_____________________________________

