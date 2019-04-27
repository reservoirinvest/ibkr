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
a = assign_var('nse')
for v in a:
    exec(v)

#_____________________________________

# p_nses.py

def p_nses(ib):
    '''Pickles nses underlying
    Arg: (ib) as connection object
    Returns: DataFrame of symbol, lot and margin with underlying info'''

    #...get the lots
    url = 'https://www.nseindia.com/content/fo/fo_mktlots.csv'
    req = requests.get(url)
    data = StringIO(req.text)
    lots_df = pd.read_csv(data)
    lots_df = lots_df[list(lots_df)[1:3]]
    lots_df.columns = ['symbol', 'lotsize']

    # clean up symbols and remove nan
    lots_df = lots_df.assign(lotsize=pd.to_numeric(lots_df.iloc[:, 1], 
                                                   errors='coerce')).dropna()
    lots_df.symbol = lots_df.symbol.str.strip()

    # convert to dictionary
    d = lots_df.to_dict('index').values()
    li =[list(d1.values()) for d1 in d]
    lots_dict = {l[0]: int(l[1]) for l in li}

    #...get the margins
    tp = pd.read_html('https://www.tradeplusonline.com/Equity-Futures-Margin-Calculator.aspx')
    df_tp = tp[1][2:].iloc[:, :-1]
    df_tp = df_tp.iloc[:, [0,5]]
    df_tp.columns=['nseSymbol', 'margin']

    df_tp.margin = df_tp.margin.apply(pd.to_numeric, errors='coerce', downcast='integer') # convert margin to numeric

    df_tp = df_tp.assign(lot=df_tp.nseSymbol.map(lots_dict))

    df_slm = df_tp.reset_index(drop=True)

    # Truncate to 9 characters for ibSymbol
    df_slm['ibSymbol'] = df_slm.nseSymbol.str.slice(0,9)

    # nseSymbol to ibSymbol dictionary for conversion
    ntoi = {'M&M': 'MM', 'M&MFIN': 'MMFIN', 'L&TFH': 'LTFH', 'NIFTY': 'NIFTY50'}

    # remap ibSymbol, based on the dictionary
    df_slm.ibSymbol = df_slm.ibSymbol.replace(ntoi)
    discards = ['NIFTYMID5', 'NIFTYIT', 'LUPIN']
    df_slm = df_slm[~df_slm.nseSymbol.isin(discards)]

    # !!!****DATA LIMITER***
    # Note: Symbols of ALL existing postions to be in the list to prevent remqty from failing!!
#     df_slm = df_slm[df_slm.ibSymbol.isin(['PNB', 'JSWSTEEL', 'BANKNIFTY'])]
    #________________________

    # separate indexes and equities, eliminate discards from df_slm
    indexes = ['NIFTY50', 'BANKNIFTY']

    equities = sorted([s for s in df_slm.ibSymbol if s not in indexes+discards])

    symbols = equities+indexes

    cs = [Stock(s, exchange) if s in equities else Index(s, exchange) for s in df_slm.ibSymbol]

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
    df_clmu = pd.DataFrame.from_dict(qcs_dict, orient='index', columns=['contract']).\
             join(pd.DataFrame.from_dict(lots_dict, orient='index', columns=['lot'])).\
             join(pd.DataFrame.from_dict(margins, orient='index', columns=['margin'])).\
             join(pd.DataFrame.from_dict(undPrices, orient='index', columns=['undPrice']))

    df_clmu = df_clmu.dropna()  # to remove discards

    df_clmu.columns=['contract', 'lot', 'margin', 'undPrice']
    df_clmu.to_pickle(fspath+'_lot_margin.pickle')
    
    return df_clmu

#_____________________________________

# p_nseopts.py

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
    #     ib.sleep(1) # to get the tickers filled

    df_opt1 = df_opt1.assign(optPrice = [t.marketPrice() for t in opt_tickers])

    df_opt1 = df_opt1[df_opt1.optPrice > 0.0]

    cols=['symbol', 'expiration', 'strike']
    df_opt2 = pd.merge(df4, df_opt1, on=cols).drop('cid', 1).reset_index(drop=True)

    # Get lotsize and margin for the underlying symbol
    df_opt2 = df_opt2.assign(lotsize = lotsize)
    df_opt2 = df_opt2.assign(optMargin = margin)

    opt_contracts = [opt_iDict[i] for i in df_opt2.optId]

    df_opt2 = df_opt2.assign(rom=df_opt2.optPrice*df_opt2.lotsize/df_opt2.optMargin*365/df_opt2.dte).sort_values('rom', ascending=False)

    # arrange the columns
    cols = ['optId', 'symbol', 'right', 'expiration', 'dte', 'strike', 'undPrice', 
            'lo52', 'hi52', 'Fall', 'Rise', 'loFall', 'hiRise', 'Std', 'loStd', 'hiStd', 
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
    df_und = pd.read_pickle(fspath+'_lot_margin.pickle').dropna()

    df_und = df_und.assign(und_remq=
                           (nse_assignment_limit/(df_und.lot*df_und.undPrice)).astype('int')) # remaining quantities in entire nse

    # from portfolio
    #_______________

    p = util.df(ib.portfolio()) # portfolio table

    # extract option contract info from portfolio table
    dfp = pd.concat([p, util.df([c for c in p.contract])[util.df([c for c in p.contract]).columns[:7]]], axis=1).iloc[:, 1:]
    dfp = dfp.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})

    # join the position with underlying contract details
    dfp1 = dfp.set_index('symbol').join(df_und, how='left').drop(['contract'], axis=1)
    dfp1 = dfp1.assign(qty=(dfp1.position/dfp1.lot).astype('int'))

    dfp1.loc[dfp1.und_remq == 0, 'und_remq'] = 1   # for very large priced shares such as AMZN, BKNG, etc

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
                                       get_prec((df.optPrice*minRom/df.rom), 0.05), 
                                       df.optPrice+0.05))

    df.loc[df.expPrice < minOptPrice, 'expPrice'] = minOptPrice # set to minimum optPrice
    
    df = df.assign(expRom=df.expPrice*df.lotsize/df.optMargin*365/df.dte). \
          sort_values('rom', ascending=False) # calculate the expected rom
    
    df = grp_opts(df)
    
    df.to_pickle(fspath+'_targetopts.pickle') # pickle the targets

    return df

#_____________________________________

# target_nse.py

def target_nse(ib):
    '''Generates a target of naked options
    Arg: (ib) as a connection object
    Returns (df) a DataFrame of targets and pickles them'''
    
    # get remaining quantities
    dfrq = remqty_nse(ib) # remaining quantities df

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

# nse_tgt_update.py

def nse_tgt_update(ib):
    '''Dynamically update target dataframe (2 mins for 400 rows) for:
        - remaining quantity
        - price
    Arg: (ib) as connection object
    Returns:
       dft as DataFrame
       And writes price and remq to _targetopts.pickle
    '''

    dft = pd.read_pickle(fspath+'_targetopts.pickle')

    # update remaining quantities
    dfr = remqty_nse(ib)
    dfr_dict = dfr.remqty.to_dict()
    dft1 = dft.assign(remqty=dft.symbol.map(dfr_dict)).reset_index()

    # update price and rom
    dft1 = upd_opt(ib, dft1)
    
    # generate expPrice
    return gen_expPrice(ib, dft)

#_____________________________________

# nse_process.py

# Weekend update functions
# __________________________

def nse_weekend_process(ib):
    '''Weekend process to generate underlyings (lot+margin) and option pickles
    Arg: (ib) as connection
    Returns: 
        None
        But pickles to _lot_margin.pickle & (SYMBOL).pkl'''
    
    util.logToFile(logpath+'_weekend.log')  # create log file
    with open(logpath+'_weekend.log', 'w'): # clear the previous log
        pass
    
    # Step # 1: generate the underlyings
    dfu = p_nses(ib).dropna()

    # Step # 2: generate the option pickles
    [p_nseopts(ib=ib,undContract=u, undPrice=p, lotsize=z, margin=m) 
     for u, p, z, m in zip(dfu.contract, dfu.undPrice, dfu.lot, dfu.margin)]
    
    return None
    
def nse_pickle_remaining_opts(ib, keep_pickles = True):
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
        [p_nseopts(ib=ib,undContract=u, undPrice=p, lotsize=z, margin=m)
         for u, p, z, m in zip(dfr.contract, dfr.undPrice, dfr.lot, dfr.margin)]
    else:
        [p_nseopts(ib=ib,undContract=u, undPrice=p, lotsize=z, margin=m)
         for u, p, z, m in zip(dfu.contract, dfu.undPrice, dfu.lot, dfu.margin)]
        
    return None

# Everyday update function
# ________________________

def nse_everyday_process(ib):
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
    upd_nses(ib, dfu)
    
    # ...update option prices
    fs = listdir(fspath)
    
    # collect the options
    optsList = [f for f in fs if f[-3:] == 'pkl']
    dfopts = pd.concat([pd.read_pickle(fspath+f)
                     for f in optsList], axis=0, sort=True).reset_index(drop=True)

    # update the option prices
    df = upd_opt(ib, dfopts)
    
    # ...generate targets and pickle, with new remaining quantities
    dft = target_nse(ib)
    
    return None

# Dynamic update function
# _______________________

def nse_dynamic_update(ib):
    '''dynamically update the target with price and remaining quantities
    Arg: (ib) as connection object
    Returns: dft as DataFrame and re-pickles it
       '''
    util.logToFile(logpath+'_dynamic.log')  # create log file
    
    return nse_tgt_update(ib)

#_____________________________________

