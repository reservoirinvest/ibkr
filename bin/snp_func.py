# imports.py
import pandas as pd
import requests
from io import StringIO
from itertools import product, repeat
from os import listdir
import logging

from ib_insync import *

from helper import *

# do the assignments from JSON
a = assign_var('snp')
for v in a:
    exec(v)

#_____________________________________

# p_snps.py

def p_snps(ib):
    '''Pickles snps underlying (1 minute)
    Arg: (ib) as connection object
    Returns: Dataframe of symbol, lot, margin with underlying info'''

    # exclusion list
    excl = ['VXX','P', 'TSRO', 'GOOGL']

    # Download cboe weeklies to a dataframe
    dls = "http://www.cboe.com/publish/weelkysmf/weeklysmf.xls"

    snp100 = list(pd.read_html('https://en.wikipedia.org/wiki/S%26P_100', 
                               header=0, match='Symbol')[0].loc[:, 'Symbol'])
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
    stocks = [Stock(symbol=s, exchange=exchange, currency='USD') for s in symbols]

    stkblks = [stocks[i: i+blk] for i in range(0, len(stocks), blk)] # blocks of stocks

    # qualify the contracts
    contracts = [ib.qualifyContracts(*s) for s in stkblks]
    contracts = [contract for subl in contracts for contract in subl]
    qcs_dict = {q.symbol: q for q in contracts}

    # get the margins
    orders = [Order(action='SELL', orderType='MKT', totalQuantity=100, whatIf=True)]*len(contracts)
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

    df_clmu = df_clmu.assign(margin=abs(pd.to_numeric(df_clmu.margin)).astype('int')) # convert to int

    df_clmu.columns=['contract', 'margin', 'undPrice', 'lot']
    
    df_clmu = df_clmu.sort_index()
    
    df_clmu.to_pickle(fspath+'_lot_margin.pickle')
    
    return df_clmu

#_____________________________________

# p_snpopts.py

def p_snpopts(ib, undContract, undPrice, lotsize=100):
    '''Pickles the option chains
    Args:
        (ib) ib connection as object
        (undContract) underlying contract as object
        (undPrice) underlying contract price as float
        (margin) margin of undContract (used as a surrogate for option)
        (lotsize) lot-size as float is 100'''
    
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

    Std = {d: get_rollingmax_std(ib, c, d, tradingdays) for c, d in zip(repeat(undContract), dtes)}

    df4['Std'] = df4.dte.map(Std)

    df4['loStd'] = df4.undPrice - df4['Std'].multiply(putstdmult)
    df4['hiStd'] = df4.undPrice + df4['Std'].multiply(callstdmult)

    # flter puts and calls by standard deviation
    df_puts = df4[df4.strike < df4.loStd]
    df_calls = df4[df4.strike > df4.hiStd]

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

    # get the price
    df_opt1 = df_opt1.assign(optPrice = [t.marketPrice() for t in opt_tickers])
    df_opt1 = df_opt1[df_opt1.optPrice > 0.0]

    # get the margins
    opts = df_opt1.optId.map(opt_iDict)
    orders = [Order(action='SELL', orderType='MKT', totalQuantity=1, whatIf=True)]*len(opts)
    margins = [ib.whatIfOrder(c, o).initMarginChange for c, o in zip(opts, orders)]
    df_opt1 = df_opt1.assign(optMargin = margins)
    df_opt1 = df_opt1.assign(optMargin = abs(pd.to_numeric(df_opt1.optMargin).astype('int'))) # convert to int

    cols=['symbol', 'expiration', 'strike']
    df_opt2 = pd.merge(df4, df_opt1, on=cols).drop('cid', 1).reset_index(drop=True)

    # Get lotsize and margin for the underlying symbol
    df_opt2 = df_opt2.assign(lotsize = lotsize)

    df_opt2 = df_opt2.assign(rom=df_opt2.optPrice*df_opt2.lotsize/df_opt2.optMargin*365/df_opt2.dte).sort_values('rom', ascending=False)
    
    # arrange the columns
    cols = ['optId', 'symbol', 'right', 'expiration', 'dte', 'strike', 'undPrice', 
            'lo52', 'hi52', 'Fall', 'Rise', 'loFall', 'hiRise', 'Std', 'loStd', 'hiStd', 
            'lotsize', 'optPrice', 'optMargin', 'rom']

    df_opt2 = df_opt2[cols]
    
    df_opt2.to_pickle(fspath+symbol+'.pkl')
    
    return None

#_____________________________________

# upd_snps.py

def upd_snps(ib, dfu):
    '''Updates the underlying snps
    Args:
       (ib) as connection object
       (dfu) as the underlying dataframe from p_snps
    Returns: None
       pickles back DataFrame with updated undPrice and margin'''

    # update prices
    tickers = ib.reqTickers(*dfu.contract)
    undPrices = {t.contract.symbol: t.marketPrice() for t in tickers} # {symbol: undPrice}

    # update margins - based on earliest expiration and strike closest to underlying price
    chains = {c.symbol: ib.reqSecDefOptParams(underlyingConId=c.conId, underlyingSecType=c.secType, underlyingSymbol=c.symbol, futFopExchange='')[0] for c in dfu.contract}

    lots_dict = dfu.lot.to_dict()

    order = Order(action='SELL', orderType='MKT', totalQuantity=100, whatIf=True)

    mdict = {i[0].symbol: abs(int(pd.to_numeric(ib.whatIfOrder(i[0], i[1]).initMarginChange))) for i in zip((c for c in dfu.contract), repeat(order))}

    # updates
    dfu['undPrice'].update(pd.Series(undPrices))
    dfu['margin'].update(pd.Series(mdict))    

    # writeback
    dfu.to_pickle(fspath+'_lot_margin.pickle')
    
    return dfu

#_____________________________________

# remqty_snp.py

def remqty_snp(ib):
    '''generates the remaining quantities dictionary
    Args:
        (ib) as connection object
    Returns:
        remqty as a dictionary of {symbol: remqty}
        '''
    df_und = pd.read_pickle(fspath+'_lot_margin.pickle').dropna()

    df_und = df_und.assign(und_remq=
                           (snp_assignment_limit/(df_und.lot*df_und.undPrice)).astype('int')) # remaining quantities in entire snp

    df_und = pd.read_pickle(fspath+'_lot_margin.pickle').dropna()

    df_und = df_und.assign(und_remq=
                           (snp_assignment_limit/(df_und.lot*df_und.undPrice)).astype('int')) # remaining quantities in entire snp

    df_und.loc[df_und.und_remq == 0, 'und_remq'] = 1   # for very large priced shares such as AMZN, BKNG, etc

    # from portfolio
    #_______________

    p = util.df(ib.portfolio()) # portfolio table

    # extract option contract info from portfolio table
    dfp = pd.concat([p, util.df([c for c in p.contract])[util.df([c for c in p.contract]).columns[:7]]], axis=1).iloc[:, 1:]
    dfp = dfp.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})

    # join the position with underlying contract details
    dfp1 = dfp.set_index('symbol').join(df_und, how='left').drop(['contract'], axis=1)
    dfp1 = dfp1.assign(qty=(dfp1.position).astype('int'))



    # get the remaining quantities
    remqty_dict = (dfp1.groupby(dfp1.index)['qty'].sum()+
                   dfp1.groupby(dfp1.index)['und_remq'].mean()).to_dict()

    remqty_dict = {k:(v if v > 0 else 0) for k, v in remqty_dict.items()} # portfolio's remq with negative values removed

    df_und1 = df_und.assign(remqty=df_und.index.map(remqty_dict))

    df_und1 = df_und1.assign(remqty=df_und1.remqty.fillna(df_und1.und_remq).astype('int'))
    
    return df_und1

#_____________________________________

#gen_expPrice.py

def gen_expPrice(ib, df):
    '''generates expected price
    Args: 
        (ib) as the connection object
        (df) as the target options from _targetopt.pickle
    Returns:
        updated df with new recalculated expPrice and expRom
        And pickles the target options
    '''
    
    df = df.assign(expPrice=np.where(df.rom < minRom, 
                                   get_prec((df.optPrice*minRom/df.rom), 0.01), 
                                   df.optPrice*1.1))

    df.loc[df.expPrice < minOptPrice, 'expPrice'] = minOptPrice # set to minimum optPrice

    df = df.assign(expRom=df.expPrice*df.lotsize/df.optMargin*365/df.dte). \
          sort_values('rom', ascending=False) # calculate the expected rom
    
    df = grp_opts(df)
    
    df.to_pickle(fspath+'_targetopts.pickle') # pickle the targets

    return df

#_____________________________________

# target_snp.py

def target_snp(ib):
    '''Generates a target of naked options
    Arg: (ib) as a connection object
    Returns (df) a DataFrame of targets and pickles them'''
    
    # get remaining quantities
    dfrq = remqty_snp(ib) # remaining quantities df

    cols = ['optId', 'symbol', 'right', 'expiration', 'dte', 'strike', 'undPrice', 
    'lo52', 'hi52', 'Fall', 'Rise', 'loFall', 'hiRise', 'Std', 'loStd', 'hiStd', 
    'lotsize', 'optPrice', 'optMargin', 'rom']

    fs = listdir(fspath)

    optsList = [f for f in fs if f[-3:] == 'pkl']

    df1 = pd.concat([pd.read_pickle(fspath+f) 
                     for f in optsList], axis=0, sort=True).reset_index(drop=True)[cols]

    # filter for high probability and margin
    df2 = df1[((df1.strike > df1.hi52) | 
               (df1.strike < df1.lo52))].sort_values('rom', ascending=False)
    df2 = df2[df2.optMargin < 1e+308]  # Remove very large number in margin (1.7976931348623157e+308)

    df2 = df2.assign(remqty=df2.symbol.map(dfrq.remqty.to_dict()))   # remaining quantities
    df2 = df2[df2.remqty > 0].reset_index(drop=True) # remove blacklisted (remqty = 0)

    df2 = df2.groupby('symbol').head(nLargest) # get the top 5

    # generate expPrice
    return gen_expPrice(ib, df2)

#_____________________________________

# snp_tgt_update.py

def snp_tgt_update(ib):
    '''Dynamically update target dataframe (2 mins for 400 rows) for:
        - remaining quantity
        - price
    Arg: (ib) as connection object
    Returns:
       targetopts DataFrame with updated price and rom 
       And writes price and remq to _targetopts.pickle
    '''

    dft = pd.read_pickle(fspath+'_targetopts.pickle')

    # update remaining quantities
    dfr = remqty_snp(ib)
    dfr_dict = dfr.remqty.to_dict()
    df2 = dft.assign(remqty=dft.symbol.map(dfr_dict)).reset_index()

    # update price and rom
    df2 = upd_opt(ib, df2)
    
    # generate expPrice
    return gen_expPrice(ib, dft)

#_____________________________________

# snp_process.py

# Weekend update functions
# __________________________

def snp_weekend_process(ib):
    '''Weekend process to generate underlyings (lot+margin) and option pickles
    Arg: (ib) as connection
    Returns: 
        None
        But pickles to _lot_margin.pickle & (SYMBOL).pkl'''
    
    util.logToFile(logpath+'_weekend.log')  # create log file
    with open(logpath+'_weekend.log', 'w'): # clear the previous log
        pass
    
    # Step # 1: generate the underlyings
    dfu = p_snps(ib).dropna()

    # Step # 2: generate the option pickles
    [p_snpopts(ib=ib,undContract=u, undPrice=p) 
     for u, p in zip(dfu.contract, dfu.undPrice)]
    
    return None
    
def snp_pickle_remaining_opts(ib, keep_pickles = True):
    '''Upon any failure, pickle remaining options upon failure
    Arg: 
        (ib) as connection
        <keep_pickles> as True / False for keeping existing pickles
    Returns:
       None'''
    
    util.logToFile(logpath+'_weekend.log')  # create log file
    
    # get the underlying dataframe
    dfu = pd.read_pickle(fspath+'_lot_margin.pickle')
    
    if keep_pickles:
        fs = listdir(fspath)
        optsList = [f[:-4] for f in fs if f[-3:] == 'pkl']
        dfr = dfu[~dfu.index.isin(optsList)] # remaining dfs (without existing pickles)
        [p_snpopts(ib=ib,undContract=u, undPrice=p)
         for u, p in zip(dfr.contract, dfr.undPrice)]
    else:
        [p_snpopts(ib=ib,undContract=u, undPrice=p)
         for u, p in zip(dfu.contract, dfu.undPrice)]
        
    return None

# Everyday update function
# ________________________

def snp_everyday_process(ib):
    '''everyday process to update update underlyings, option prices and generate targets
    (takes about 40 mins)
    Arg: (ib) as connection
    Returns:
       None
       But pickles lot and margin updates to underlyings
    '''
    util.logToFile(logpath+'_everyday.log')  # create log file
    
    # ...update underlyings
    dfu = pd.read_pickle(fspath+'_lot_margin.pickle').dropna()
    upd_snps(ib, dfu)
    
    # ...update option prices
    fs = listdir(fspath)
    
    # collect the options
    optsList = [f for f in fs if f[-3:] == 'pkl']
    dfopts = pd.concat([pd.read_pickle(fspath+f)
                     for f in optsList], axis=0, sort=True).reset_index(drop=True)

    # update the option prices
    df = upd_opt(ib, dfopts)
    
    # ...generate targets and pickle, with new remaining quantities
    dft = target_snp(ib)
    
    return None

# Dynamic update function
# _______________________

def snp_dynamic_update(ib):
    '''dynamically update the target with price and remaining quantities
    Arg: (ib) as connection object
    Returns: dft as DataFrame and re-pickles it
       '''
    util.logToFile(logpath+'_dynamic.log')  # create log file
    
    return snp_tgt_update(ib)

#_____________________________________

