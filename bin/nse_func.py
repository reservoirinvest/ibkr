# imports.py
import pandas as pd
import requests
from io import StringIO
from itertools import product, repeat
from os import listdir
import logging
from bs4 import BeautifulSoup
import csv

from ib_insync import *

from helper import *

# do the assignments from JSON
a = assign_var('nse')
for v in a:
    exec(v)

#_____________________________________

# base.py

def base(ib):
    '''Creates the base.pickle with qualified underlyings
    Arg: (ib) as connection object
    Returns: None'''

    #...make an expiry date dictionary

    res = requests.get('https://www.nseindia.com/live_market/dynaContent/live_watch/fomwatchsymbol.jsp?key=NIFTY&Fut_Opt=Futures')
    soup = BeautifulSoup(res.content, 'lxml')
    table = soup.find_all('table')
    df_exp = pd.read_html(str(table[1]), header=0)[0] # to dataframe
    dt = pd.to_datetime(df_exp['Expiry Date'])
    exp_dates = {d.strftime('%b-%y').lower(): d.strftime('%Y%m%d') for d in dt}

    #...get the lots

    # set up the dates
    url = 'https://www.nseindia.com/content/fo/fo_mktlots.csv'
    req = requests.get(url)
    data = StringIO(req.text)
    lots_df = pd.read_csv(data)
    lots_df = lots_df[list(lots_df)[1:5]]

    # strip whitespace from columns and make it lower case
    lots_df.columns = lots_df.columns.str.strip().str.lower() 

    # strip all string contents of whitespaces
    lots_df = lots_df.applymap(lambda x: x.strip() if type(x) is str else x)

    # regenerate the columns using expiry date dictionary
    lots_df.columns = ['symbol'] + list(lots_df.set_index('symbol').columns.map(exp_dates))

    # remove symbols
    lots_df = lots_df[~lots_df.symbol.isin(['Symbol'])]

    # force to appropriate int / float
    lots_df = lots_df.apply(pd.to_numeric, errors='ignore')

    df_l = pd.melt(lots_df.iloc[:, [0,1,2,3]], id_vars=['symbol'], var_name='expiration', value_name='lot').dropna()
    df_l = df_l.assign(dte=[get_dte(e) for e in df_l.expiration])

    df_l = df_l.assign(ibSymbol=df_l.symbol.str.slice(0,9)).drop('symbol', axis=1) # for ibSymbol and drop symbol

    # make the lots into int64 for consistency
    df_l = df_l.assign(lot=df_l.lot.astype('int64'))

    # nseSymbol to ibSymbol dictionary for conversion
    ntoi = {'M&M': 'MM', 'M&MFIN': 'MMFIN', 'L&TFH': 'LTFH', 'NIFTY': 'NIFTY50'}

    # remap ibSymbol, based on the dictionary
    df_l.ibSymbol = df_l.ibSymbol.replace(ntoi)

    # remove unnecessary symbols
    discards = ['LUPIN']
    df_l = df_l[~df_l.ibSymbol.isin(discards)].reset_index(drop=True)

    # get the underlying contracts qualified
    indexes = {'BANKNIFTY': 'IND', 'NIFTY50': 'IND', 'NIFTYIT': 'IND'}

    # qualify the underlyings
    cs = [Index(s, exchange) if s in indexes.keys() else Stock(s, exchange) for s in df_l.ibSymbol.unique()]
    qcs = ib.qualifyContracts(*cs) # qualified underlyings
    qcs = [q for c in qcs for q in ib.qualifyContracts(Contract(conId=c.conId))] # to get secType info
    qcs_dict = {q.symbol: q for q in qcs}

    # get the underlying conId
    undId = {q.symbol: q.conId for q in qcs}
    df_l = df_l.assign(undId=df_l.ibSymbol.map(undId))

    # get the underlying prices
    tickers = ib.reqTickers(*qcs)
    undPrices = {t.contract.symbol: t.marketPrice() for t in tickers} # {symbol: undPrice}

    df_l = df_l.assign(undPrice=df_l.ibSymbol.map(undPrices))

    # get max fall-rise
    mfr = [get_maxfallrise(ib, c, d) for c, d in zip(df_l.ibSymbol.map(qcs_dict), df_l.dte)]

    df_mfr = pd.DataFrame(mfr, columns=['lo52', 'hi52', 'Fall', 'Rise'])
    df_lm = df_l.join(df_mfr)

    # get the standard deviation
    sd = [get_rollingmax_std(ib, c, d, tradingdays) for c, d in zip(df_l.ibSymbol.map(qcs_dict), df_l.dte)]

    df_sd = pd.DataFrame(sd, columns=['Std'])
    df_lms = df_lm.join(df_sd)

    #...get the strikes from option chains

    ch_list = [(q.symbol, 'IND', q.conId) 
               if q.symbol in indexes 
               else (q.symbol, 'STK', q.conId) 
               for q in qcs]

    chains = {s: ib.reqSecDefOptParams(underlyingSymbol=s, underlyingSecType=t, underlyingConId=c, futFopExchange='') for s, t, c in ch_list}

    strikes = {k: v[0].strikes for k, v in chains.items()}

    df_ls = df_lms.assign(strikes=df_lms.ibSymbol.map(strikes))

    s = (pd.DataFrame(df_ls.pop('strikes').values.tolist(), index=df_ls.index)
            .stack()
            .rename('strike')
            .reset_index(level=1, drop=True))

    df_ls = df_ls.join(s).reset_index(drop=True)

    df_ls['loStd'] = df_ls.undPrice - df_ls['Std'].multiply(putstdmult)
    df_ls['hiStd'] = df_ls.undPrice + df_ls['Std'].multiply(callstdmult)

    # flter puts and calls by standard deviation
    df_puts = df_ls[df_ls.strike < df_ls.loStd]
    df_calls = df_ls[df_ls.strike > df_ls.hiStd]

    # with rights
    df_puts = df_puts.assign(right='P')
    df_calls = df_calls.assign(right='C')

    df_opt = pd.concat([df_puts, df_calls]).reset_index(drop=True)

    df_opt.to_pickle(fspath+'base.pickle')
    
    return None

#_____________________________________

# opts.py

def opts(ib):
    '''Pickles the option chains
    Args: (ib) as connection object
    Returns: None. But pickles to opts.pickle'''
    
    df = pd.read_pickle(fspath+'base.pickle')

    df = df[(df.dte > mindte) & (df.dte < maxdte)].reset_index(drop=True)  # limiting dtes

    optipl = [Option(s, e, k, r, 'NSE') for s, e, k, r in zip(df.ibSymbol, df.expiration, df.strike, df.right)]

    optblks = [optipl[i: i+blk] for i in range(0, len(optipl), blk)] # blocks of optipl

    # qualify the contracts
    cblks = [ib.qualifyContracts(*s) for s in optblks]

    contracts = [z for x in cblks for z in x]

    # prepare the qualified options dataframe
    dfq = util.df(contracts).iloc[:, 1:6]
    dfq.columns=['optId', 'ibSymbol', 'expiration', 'strike', 'right'] # rename columns
    dfq = dfq.set_index(['ibSymbol', 'expiration', 'strike', 'right']) # set index

    # filter options who have conId
    df_opt = df.set_index(['ibSymbol', 'expiration', 'strike', 'right']).join(dfq).dropna().reset_index()

    # convert optId to int for price and margin
    df_opt = df_opt.assign(optId=df_opt.optId.astype('int32'))

    # get the prices
    opt_tickers = ib.reqTickers(*contracts)

    optPrices = {t.contract.conId: t.marketPrice() for t in opt_tickers}

    df_opt = df_opt.assign(optPrice=df_opt.optId.map(optPrices))
    df_opt = df_opt[~(df_opt.optPrice <= 0)].reset_index(drop=True) # remove options without a price

    # ... get the margins and rom

    # get the closest strike
    df_cstrike = df_opt.iloc[df_opt.groupby('ibSymbol').apply(lambda df: abs(df.strike - df.undPrice).idxmin())]

    # prepare the contracts and orders
    cdict = {c.conId: c for c in contracts} # {conId: contract}
    contracts = df_cstrike.optId.map(cdict)
    orders = [Order(action='SELL', orderType='MKT', totalQuantity=lot, whatIf=True) for lot in df_cstrike.lot]

    mdict = {c.symbol: ib.whatIfOrder(c, o).initMarginChange for c, o in zip(contracts, orders)}

    # Remove very large number in margin (1.7976931348623157e+308)
    mdict = {key: value for key, value in mdict.items() if float(value) < 1e7}

    # append the margins
    df_opt = df_opt.assign(margin=df_opt.ibSymbol.map(mdict).astype('float'))

    # calculate rom
    df_opt = df_opt.assign(rom=df_opt.optPrice/df_opt.margin*365/df_opt.dte*df_opt.lot)
    
    # rename symbols column
    df_opt = df_opt.rename(columns={'ibSymbol': 'symbol'})

    df_opt.to_pickle(fspath+'opts.pickle')
    
    return None

#_____________________________________

# remqty_nse.py

def remqty_nse(ib):
    '''generates the remaining quantities dictionary
    Args:
        (ib) as connection object
    Returns:
        dfrq as a datframe with symbol and remaining quantities
        '''
    # from portfolio
    #_______________

    p = util.df(ib.portfolio()) # portfolio table

    # extract option contract info from portfolio table
    dfp = pd.concat([p, util.df([c for c in p.contract])[util.df([c for c in p.contract]).columns[:7]]], axis=1).iloc[:, 1:]
    dfp = dfp.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})

    # get the total position
    dfp1 = dfp.groupby('symbol').sum()['position']

    # from options pickle
    #____________________

    # get the options
    df_opt = pd.read_pickle(fspath + 'opts.pickle')
    df_opt = df_opt.assign(und_remq=(nse_assignment_limit/(df_opt.lot*df_opt.undPrice)).astype('int')) # remaining quantities in entire nse

    # compute the remaining quantities
    df_opt1 = df_opt.groupby('symbol').first()[['lot', 'margin', 'und_remq']]

    df_opt2 = df_opt1.join(dfp1).fillna(0).astype('int')
    df_opt2 = df_opt2.assign(remqty=df_opt2.und_remq+(df_opt2.position/df_opt2.lot).astype('int'))

    dfrq = df_opt2[['remqty']]

    return dfrq

#_____________________________________

# targets.py

def targets(ib):
    '''Generates a target of naked options
    Arg: (ib) as a connection object
    Returns (df) a DataFrame of targets with expPrice and pickles them'''
    
    # get remaining quantities
    dfrq = remqty_nse(ib) # remaining quantities df

    df1 = pd.read_pickle(fspath+'opts.pickle')

    # filter for high probability and margin
    df2 = df1[((df1.strike > df1.hi52) | 
               (df1.strike < df1.lo52))].sort_values('rom', ascending=False)

    df2 = df2[df2.margin < 1e+308]  # Remove very large number in margin (1.7976931348623157e+308)

    df2 = df2.assign(remqty=df2.symbol.map(dfrq.remqty.to_dict()))   # remaining quantities
    df2 = df2[df2.remqty > 0].reset_index(drop=True) # remove blacklisted (remqty <= 0)

    df2 = df2.groupby('symbol').head(nLargest) # get the top 3

    # generate expPrice
    return gen_expPrice(ib, df2)

#_____________________________________

# gen_expPrice.py

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
    
    df = df.assign(expRom=df.expPrice*df.lot/df.margin*365/df.dte). \
          sort_values('rom', ascending=False) # calculate the expected rom
    
    df = df.dropna().reset_index(drop=True)
    
    # Establish sowing quantities
    putFall = (df.right == 'P') & ((df.undPrice - df.strike) > df.Fall)
    callRise = (df.right == 'C') & ((df.strike - df.undPrice) > df.Rise)

    df = df.assign(qty=np.where(callRise | putFall, df.remqty, 1))
    
    df = grp_opts(df)
    
    df.to_pickle(fspath+'targets.pickle') # pickle the targets

    return df

#_____________________________________

# upd.py

def upd(ib, dfopt):
    '''Updates the nse options' price and margin
    Takes 3 mins for 450 options
    Args:
       (ib) as connection object
       (dfopt) as the dataframe from targets.pickle
    Returns: None
       pickles back DataFrame with updated undPrice and margin'''
    
    # get the contracts
    cs=[Contract(conId=c) for c in dfopt.optId]

    blks = [cs[i: i+blk] for i in range(0, len(cs), blk)]
    cblks = [ib.qualifyContracts(*s) for s in blks]
    contracts = [z for x in cblks for z in x]

    # update prices
    tickers = ib.reqTickers(*contracts)
    optPrices = {t.contract.conId: t.marketPrice() for t in tickers} # {symbol: undPrice}

    dfopt = dfopt.assign(optPrice = dfopt.optId.map(optPrices))

    # get the margins
    orders = [Order(action='SELL', orderType='MKT', totalQuantity=lot, whatIf=True) for lot in dfopt.lot]

    mdict = {c.conId: ib.whatIfOrder(c, o).initMarginChange for c, o in zip(contracts, orders)} # making dictionary takes time!

    # Remove very large number in margin (1.7976931348623157e+308)
    mdict = {key: value for key, value in mdict.items() if float(value) < 1e7}

    # assign the margins
    dfopt = dfopt.assign(margin=dfopt.optId.map(mdict).astype('float'))

    # calculate rom
    dfopt = dfopt.assign(rom=dfopt.optPrice/dfopt.margin*365/dfopt.dte*dfopt.lot)

    # regenerate expected price
    df = gen_expPrice(ib, dfopt)

    df.to_pickle(fspath+'targets.pickle')
    
    return None

#_____________________________________

# watchlists.py

def watchlists(ib):
    '''Generate watchlist
       First with existing positions
       Then with sowing symbols
    Arg: (ib) as connection object
    Returns: None. Generates watch.csv and targets.csv'''

    # get the portfolio
    p = ib.portfolio()

    # get the targets
    df = pd.read_pickle(fspath+'targets.pickle')

    # make the symbol list
    sy = list(util.df(list(util.df(p)['contract'])).symbol.unique()) + \
         list(df.symbol.unique())

    # make and pickle the watchlist
    watch = [('DES', s, 'STK', 'NSE') if s not in ['NIFTY50', 'BANKNIFTY', 'NIFTYIT'] 
             else ('DES', s, 'IND', 'NSE') for s in sy]

    # write to watch.csv
    util.df(watch).to_csv(fspath+'watch.csv', header=None, index=None)
    
    # write targets to csv
    df.to_csv(fspath+'targets.csv', index=None, header=True)

#_____________________________________

