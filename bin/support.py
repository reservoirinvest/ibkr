# imports.py

"""Support programs for main
Date: 30-Sep-2019
Ver: 1.0"""

import json
import numpy as np
import datetime
import pandas as pd
from math import floor, log10
from itertools import product
import requests
from io import StringIO
import asyncio
from ib_insync import *
from os import listdir, path, unlink

#_____________________________________

# assign_var.py
def assign_var(market):
    '''Assign variables using exec
    Arg: (market) as string <'nse'>|<'snp' 
    Returns: VarList as a list of strings containing assignments
             These will be executed upon using exec()'''

    with open('variables.json', 'r') as fp:
        varDict = json.load(fp)
    
    varList = [str(k+"='"+str(v)+"'")  if type(v) is str
               else (str(k+'='+ str(v))) if type(v) is list
               else str(k+'='+str(v)) for k, v in varDict[market].items()]
    return varList

# assign common variables
for c in assign_var('common'):
    exec(c)

# assign market specific variables
for v in assign_var(market):
    exec(v)

#_____________________________________

# get_connected.py
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
                ('SNP', 'PAPER'): 1301}
    
    host = hostdict[ip]
    
    cid = 0 # initialize clientId
    max_cid = 4 # maximum clientId allowed. max possible is 32

    for i in range(cid, max_cid):
        try:
            ib = IB().connect('127.0.0.1', host, clientId=i)
            
        except Exception as e:
            print(e) # print the error
            continue # go to next
            
        break # successful try
        
    return ib

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

# get_dte.py
def get_dte(dt):
    '''Gets days to expiry
    Arg: (dt) as day in string format 'yyyymmdd'
    Returns: days to expiry as int'''
    return (util.parseIBDatetime(dt) - 
            datetime.datetime.now().date()).days

#_____________________________________

# fallrise.py
def fallrise(df_hist, dte):
    '''Gets the fall and rise for a specific dte
    Args:
        (df_hist) as a df with historical ohlc for a scrip
        (dte) as int for days to expiry
    Returns:
        {dte: {'fall': fall, 'rise': rise}} as a dictionary of floats'''
    s = df_hist.symbol.unique()[0]
    df = df_hist.set_index('date').sort_index(ascending = True)
    df = df.assign(delta = df.high.rolling(dte).max() - df.low.rolling(dte).min(), 
                        pctchange = df.close.pct_change(periods=dte))

    df1 = df.sort_index(ascending = False)
    max_fall = df1[df1.pctchange<=0].delta.max()
    max_rise = df1[df1.pctchange>0].delta.max()
    
    return (s, dte, max_fall, max_rise)

#_____________________________________

# get_prec.py
def get_prec(v, base):
    '''gives the precision value, based on base
    args:
       (v) as value needing precision in float
       (base) as the base value e.g. 0.05'''
    
    return round(round((v)/ base) * base, -int(floor(log10(base))))

#_____________________________________

# hvstPricePct.py
def hvstPricePct(dte):
    '''Gets expected price percentage from DTE for harvesting trades.
    Assumes max DTE to be 30 days.
    Arg: (dte) days to expiry as an int 
    Returns: expected harvest price percentage (xpp) as float
    Ref: http://interactiveds.com.au/software/Linest-poly.xls ... for getting curve function
    '''
#     if dte is to be extracted from contract.lastTradeDateOrContractMonth
#     dte = (util.parseIBDatetime(expiry) - datetime.datetime.now().date()).days
    
    if dte > 30:
        dte = 30  # Forces the max DTE to be 30 days
    
    xpp = 1-(103.6008 - 3.63457*dte + 0.03454677*dte*dte)/100
    
    return xpp

#_____________________________________

# sec2hms.py
def sec2hms(seconds):
    '''get a printable hh:mm:ss time of elapsed program
    Arg: (seconds) as float
    Returns: hh:mm:ss as string'''
    
    m, s = divmod(seconds,60)
    h, m = divmod(m, 60)
    
    return '{:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s))

#_____________________________________

# jup_disp_adjust.py
def jup_disp_adjust():
    '''Sets jupyter to show columns in 0.00 format and shows all columns'''
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.float_format', '{:.2f}'.format)

#_____________________________________

# delete_all_data.py
def delete_all_data(market):
    '''Deletes all data and log files
    Arg: (market) as <'NSE'> | <'SNP'> 
    Return: None'''
    
    mkt = market.lower()
    
    folderpaths = ["../data/"+mkt+"/", "../data/log/"]

    for folder in folderpaths:
        for files in listdir(folder):
            file_path = path.join(folder, files)
            try:
                if path.isfile(file_path):
                    unlink(file_path)
            except Exception as e:
#                 print(e)
                pass
                
    return None

#_____________________________________

# cancel_sells.py
def cancel_sells(ib):
    '''Cancels all sell orders
    Arg: (ib) as connection object
    Returns: [canceld_sells] list'''
    # get all the trades
    trades = ib.trades()
    
    if trades: # there is something in trades
        all_trades_df = util.df(t.contract for t in trades).join(util.df(t.orderStatus for t in trades)).join(util.df(t.order for t in trades), lsuffix='_')
        all_trades_df.rename({'lastTradeDateOrContractMonth': 'expiry'}, axis='columns', inplace=True)
        trades_cols = ['conId', 'symbol', 'localSymbol', 'secType', 'expiry', 'strike', 'right', 
                       'orderId', 'permId', 'action', 'totalQuantity', 'lmtPrice', 'status']
        trades_df = all_trades_df[trades_cols]

        # get the sell option trades which are open (SUBMITTED)
        df_open_sells = trades_df[(trades_df.action == 'SELL') & 
                  (trades_df.secType == 'OPT') &
                  trades_df.status.isin(active_status)]

        # cancel the sell open orders
        sell_openords = [t.order for t in trades if t.order.orderId in list(df_open_sells.orderId)]
        canceld_sells = [ib.cancelOrder(order) for order in sell_openords]
        
        print(f'Cancelled {len(canceld_sells)} out of {len(sell_openords)} open orders')

        return canceld_sells
    else:
        print('\nNo sells are available to cancel\n')
        return None

#_____________________________________

# get_chains.py
def get_chains(ib, market):
    '''Gets chains for the markets
    Args:
        (ib) as connection object
        (market) as <'snp'>|<'nse'>
    Returns:
        (df_chains, contracts) tuple with:
            df_chains as dataframe of symbol, expiries, undId, lots and undPrice
            contracts as the undId contracts'''
    
    if market == 'snp':
        
        # Download cboe weeklies to a dataframe
        dls = "http://www.cboe.com/publish/weelkysmf/weeklysmf.xls"

        # read from row no 11, dropna and reset index
        df_cboe = pd.read_excel(dls, header=12, 
                                usecols=[0,2,3]).loc[11:, :]\
                                .dropna(axis=0)\
                                .reset_index(drop=True)

        # remove column names white-spaces and remap to IBKR
        df_cboe.columns = df_cboe.columns.str.replace(' ', '')

        # remove '/' for IBKR
        df_cboe.Ticker = df_cboe.Ticker.str.replace('/', ' ', regex=False)

        snp100 = list(pd.read_html('https://en.wikipedia.org/wiki/S%26P_100', 
                                   header=0, match='Symbol')[0].loc[:, 'Symbol'])
        snp100 = [s.replace('.', ' ') if '.' in s else s  for s in snp100] # without dot in symbol

        # remove equities not in snp100
        df_symbols = df_cboe[~((df_cboe.ProductType == 'Equity') & ~df_cboe.Ticker.isin(snp100))]

        # rename Ticker to symbols
        df_symbols = df_symbols.rename({'Ticker': 'symbol'}, axis=1)

        # add in the lots
        df_symbols = df_symbols.assign(lot=100)

        # !!! Start DATA LIMITER !!! Get 8 symbols. 5 Equities and 3 ETFs !!!
#         df_symbols = pd.concat([df_symbols[df_symbols.ProductType == 'Equity'].head(7), df_symbols.head(3)]) # !!! DATA LIMITER !!!
        # ...!!! End DATA LIMITER !!!

        stocks = [Stock(s, exchange, currency) for s in list(df_symbols.symbol)]
        contracts = ib.qualifyContracts(*stocks)

        # get the conIds
        symConDict = {u.symbol: u.conId for u in contracts}
        conSymDict = {u.conId: u.symbol for u in contracts}

        # get the prices
        async def undpricecoro(cs):
            tasks = [ib.reqTickersAsync(c) for c in cs]
            return await asyncio.gather(*tasks)

        try:
            undpList = [u for undp in ib.run(undpricecoro(contracts), timeout = 25) for u in undp]
        except Exception as e:
            print(f"ATTENTION: The error {e} appeared while getting undPrice tickers")

        undPrices = {u.contract.symbol: u.marketPrice() for u in undpList}

        # get the chains
        async def chains_coro(und_contracts):
            '''Get the chains for underlyings
            Arg: (und_contracts) as a list
            Returns: awaits of reqSecDefOptPramsAsyncs'''
            ch_tasks = [ib.reqSecDefOptParamsAsync(underlyingSymbol=c.symbol, futFopExchange='', 
                                     underlyingConId=c.conId, underlyingSecType=c.secType)
                                     for c in und_contracts]
            return await asyncio.gather(*ch_tasks)

        ch = ib.run(chains_coro(contracts))

        chs = [b for a in ch for b in a]

        chains = {c.underlyingConId: c for c in chs}

        sek = {b for a in [list(product([k], m.expirations, m.strikes)) for k, m in chains.items()] for b in a}

        dfc = pd.DataFrame(list(sek), columns=['undId', 'expiry', 'strike'])
        dfc = dfc.assign(dte=[(util.parseIBDatetime(dt)-datetime.datetime.now().date()).days for dt in dfc.expiry])
        dfc = dfc.assign(undId=dfc.undId.astype('int32'))

        dfc = dfc.assign(symbol=dfc.undId.map(conSymDict))

        df_chains = dfc.assign(lot=100)
        df_chains = df_chains.assign(undPrice=df_chains.symbol.map(undPrices))

        df_chains = df_chains[['symbol', 'undId', 'expiry', 'dte', 'strike', 'lot', 'undPrice']]
    
    else: # for NSE
        
        url = 'https://www.nseindia.com/content/fo/fo_mktlots.csv'

        # req = requests.get(url)
        # data = StringIO(req.text)

        # In case requests does not work, use the following 
        # after commenting out the request snd data lines above
        data = 'C:\\Users\\kashir\\Downloads\\fo_mktlots.csv'
        
        lots_df = pd.read_csv(data)

        lots_df = lots_df[list(lots_df)[1:5]]

        # strip whitespace from columns and make it lower case
        lots_df.columns = lots_df.columns.str.strip().str.lower() 

        # strip all string contents of whitespaces
        lots_df = lots_df.applymap(lambda x: x.strip() if type(x) is str else x)

        # remove 'Symbol' row
        lots_df = lots_df[lots_df.symbol != 'Symbol']

        # melt the expiries into rows
        lots_df = lots_df.melt(id_vars=['symbol'], var_name='expiryM', value_name='lot').dropna()

        # remove rows without lots
        lots_df = lots_df[~(lots_df.lot == '')]

        # convert expiry to period
        lots_df = lots_df.assign(expiryM=pd.to_datetime(lots_df.expiryM, format='%b-%y').dt.to_period('M').astype('str'))

        # convert lots to integers
        lots_df = lots_df.assign(lot=pd.to_numeric(lots_df.lot, errors='coerce'))

        # convert & to %26
        lots_df = lots_df.assign(symbol=lots_df.symbol.str.replace('&', '%26'))

        # convert symbols - friendly to IBKR
        lots_df = lots_df.assign(symbol=lots_df.symbol.str.slice(0,9))
        ntoi = {'M%26M': 'MM', 'M%26MFIN': 'MMFIN', 'L%26TFH': 'LTFH', 'NIFTY': 'NIFTY50'}
        lots_df.symbol = lots_df.symbol.replace(ntoi)

        # !!! Start of DATA LIMITER !!!
        # tempsymbols = ['ACC', 'INFY', 'RELIANCE'] + ind_symbols
        # lots_df = lots_df.loc[lots_df.symbol.isin(tempsymbols), :].reset_index(drop=True)
        # ...End of DATA LIMITER !!!...

        # get the underlying prices
        symbols = lots_df.symbol.unique()

        ind_symbols = ['NIFTY50', 'BANKNIFTY', 'NIFTYIT']

        # Get the stock and index contracts
        stocks = [Stock(s, market, currency) for s in symbols if s not in ind_symbols]
        indexes = [Index(s, market, currency) for s in ind_symbols]

        stock_contracts = ib.qualifyContracts(*stocks)
        index_contracts = ib.qualifyContracts(*indexes)

        contracts = stock_contracts + index_contracts

        # get the conIds
        symConDict = {u.symbol: u.conId for u in contracts}
        conSymDict = {u.conId: u.symbol for u in contracts}

        # get the prices
        async def undpricecoro(cs):
            tasks = [ib.reqTickersAsync(c) for c in cs]
            return await asyncio.gather(*tasks)

        try:
            undpList = [u for undp in ib.run(undpricecoro(contracts), timeout = 25) for u in undp]
        except Exception as e:
            print(f"ATTENTION: The error {e} appeared while getting undPrice tickers")

        undPrices = {u.contract.symbol: u.marketPrice() for u in undpList}

        lots_df = lots_df.assign(undId=lots_df.symbol.map(symConDict), undPrice=lots_df.symbol.map(undPrices))

        # get the chains
        async def chains_coro(und_contracts):
            '''Get the chains for underlyings
            Arg: (und_contracts) as a list
            Returns: awaits of reqSecDefOptPramsAsyncs'''
            ch_tasks = [ib.reqSecDefOptParamsAsync(underlyingSymbol=c.symbol, futFopExchange='', 
                                     underlyingConId=c.conId, underlyingSecType=c.secType)
                                     for c in und_contracts]
            return await asyncio.gather(*ch_tasks)

        ch = ib.run(chains_coro(contracts))

        chs = [b for a in ch for b in a]

        chains = {c.underlyingConId: c for c in chs}

        sek = {b for a in [list(product([k], m.expirations, m.strikes)) for k, m in chains.items()] for b in a}

        dfc = pd.DataFrame(list(sek), columns=['undId', 'expiry', 'strike'])
        dfc = dfc.assign(dte=[(util.parseIBDatetime(dt)-datetime.datetime.now().date()).days for dt in dfc.expiry])
        dfc = dfc.assign(undId=dfc.undId.astype('int32'))

        dfc = dfc.assign(symbol=dfc.undId.map(conSymDict))

        dfc = dfc.assign(expiryM=pd.to_datetime(dfc.expiry).dt.strftime('%Y-%m').astype('str'))

        ix_chains = dfc.set_index(['symbol', 'undId', 'expiryM'])
        ix_lots = lots_df.set_index(['symbol', 'undId', 'expiryM'])

        df_chains = ix_chains.join(ix_lots).reset_index().drop('expiryM',1)
        
    return (df_chains, contracts)
        

#_____________________________________

# get_df_buys.py
def get_df_buys(ib, market, prec):
    '''Get the dynamic buys for latest trades
    Arg: 
        (ib) as connection object
        (market) as <'snp'> | <'nse'>
        (prec) as precision for markets <0.01> | <0.05>
    Returns: 
        df_buy as a DataFrame, ready for buy and doTrade functions'''
    #... get the open BUY trades

    # get the template
    df_opentrades = pd.read_pickle('./templates/df_opentrades.pkl')
    df_opentrades.drop(df_opentrades.index, inplace=True) # empty it!

    # get the trades
    trades = ib.openTrades()
    trade_cols = ['secType', 'conId', 'symbol', 'lastTradeDateOrContractMonth', 'strike', 'right', 'action', 
                  'status', 'orderId', 'permId', 'lmtPrice', 'filled', 'remaining']

    # append it, if available
    if trades:
        df_ot = df_opentrades.append(util.df(t.contract for t in trades).join(util.df(t.order for t in trades)).join(util.df(t.orderStatus for t in trades), lsuffix='_'))
        df_ot = df_ot[trade_cols].rename(columns={'lastTradeDateOrContractMonth': 'expiry'})
#         active_status = {'ApiPending', 'PendingSubmit', 'PreSubmitted', 'Submitted'}
        df_activebuys = df_ot[(df_ot.action == 'BUY') & (df_ot.status.isin(active_status))]
        df_activesells = df_ot[(df_ot.action == 'SELL') & (df_ot.status.isin(active_status))]
    else:
        df_activebuys = df_opentrades[trade_cols].rename(columns={'lastTradeDateOrContractMonth': 'expiry'})
        df_activesells = df_opentrades[trade_cols].rename(columns={'lastTradeDateOrContractMonth': 'expiry'})

    #... get the portfolio
    df_pfolio = portf(ib)

    # remove from portfolio options with active buys
    df_buys = df_pfolio[~df_pfolio.conId.isin(df_activebuys.conId)]

    # remove stocks, keep only options
    df_buys = df_buys[df_buys.secType == 'OPT'] 

    # Remove the opts with position > 0. These are errors / longs that should not be re-bought automatically!
    df_buys = df_buys[df_buys.position < 0]

    # Rename the column conId to optId
    df_buys = df_buys.rename(columns={'conId': 'optId'})

    # get the dte
    df_buys = df_buys.assign(dte=df_buys.expiry.apply(get_dte))

    df_buys = df_buys[df_buys.dte > 1] # spare the last day!

    #... get the expected price to sell.
    df_buys = df_buys.assign(expPrice = np.maximum( \
                                            np.minimum(df_buys.dte.apply(hvstPricePct)*df_buys.averageCost, (df_buys.marketPrice-2*prec)), \
                                            prec) \
                                            .apply(lambda x: get_prec(x, prec)))

    # set the quantity
    df_buys = df_buys.assign(qty = df_buys.position.apply(abs))
    df_buys = df_buys.assign(lot=1) # doesn't need to be lotsize for NSE
    
    return df_buys

#_____________________________________

# buy_sell_tradingblocks.py
def trade_blocks(ib, df, action, exchange):
    '''Makes SELL contract blocks for trades
    Args:
       (ib) as connection object
       (df) as the target df for setting up the trades
       (action) = <'BUY'> | <'SELL'>
       (exchange) as the market <'NSE'|'SMART'>
    Returns:
       (coblks) as contract blocks'''
    
    if exchange == 'NSE':
        sell_orders = [LimitOrder(action=action, totalQuantity=q*l, lmtPrice=expPrice) for q, l, expPrice in zip(df.qty, df.lot, df.expPrice)]
    elif exchange == 'SMART':
        sell_orders = [LimitOrder(action=action, totalQuantity=q, lmtPrice=expPrice) for q, expPrice in zip(df.qty, df.expPrice)]
    # get the contracts
    cs=[Contract(conId=c) for c in df.optId]

    blks = [cs[i: i+blk] for i in range(0, len(cs), blk)]
    cblks = [ib.qualifyContracts(*s) for s in blks]
    qc = [z for x in cblks for z in x]

    co = list(zip(qc, sell_orders))
    coblks = [co[i: i+blk] for i in range(0, len(co), blk)]
    
    return coblks

# prepares the SELL opening trade blocks
def sells(ib, df_targets, exchange):
    '''Prepares SELL trade blocks for targets
    Should NOT BE used dynamically
    Args: 
        (ib) as connection object
        (df_targets) as a dataframe of targets
        (exchange) as the exchange
    Returns: (sell_tb) as SELL trade blocks'''
    # make the SELL trade blocks
    sell_tb = trade_blocks(ib=ib, df=df_targets, action='SELL', exchange=exchange)
    
    return sell_tb

# prepare the BUY closing order trade blocks
def buys(ib, df_buy, exchange):
    '''Prepares BUY trade blocks for those without close trades.
    Can be used dynamically.
    Args:  
        (ib) as connection object
        (df_buy) as the dataframe to buy from workout
        (exchange) as the exchange
    Dependancy: sized_snp.pkl for other parameters
    Returns: (buy_tb) as BUY trade blocks'''
    
    if not df_buy.empty: # if there is some contract to close
        buy_tb = trade_blocks(ib=ib, df=df_buy, action='BUY', exchange=exchange)
    else:
        buy_tb = None
    return buy_tb

#_____________________________________

# doTrades.py
def doTrades(ib, coblks):
    '''Places trades in blocks
    Arg: 
        (ib) as connection object
        (coblks) as (contract, order) blocks'''
    trades = []
    for coblk in coblks:
        for co in coblk:
            trades.append(ib.placeOrder(co[0], co[1]))
        ib.sleep(1)
        
    return trades

#_____________________________________

# portf.py
def portf(ib):
    '''gives (fast) portfolio sorted by unrealizedPNL
    Arg: (ib) as connection object
    Returns: pf as portfolio dataframe'''
    # get the portfolio
    if ib.portfolio(): # there is something in the portfolio
        pf = util.df(ib.portfolio()).drop('account', 1)
        pc = util.df(list(pf.contract)).iloc[:, :6]
        pf = pc.join(pf.drop('contract',1)).sort_values('unrealizedPNL', ascending=True)
        pf.rename({'lastTradeDateOrContractMonth': 'expiry'}, axis='columns', inplace=True)
        dtes = {p.expiry: get_dte(p.expiry) for p in pf.itertuples() if p.secType == 'OPT'}
        pf = pf.assign(dte=pf.expiry.map(dtes))
        
        # averageCost is to be divided by 100 for SNP
        if market == 'snp':
            pf = pf.assign(averageCost=np.where(pf.secType == 'OPT', pf.averageCost/100, pf.averageCost))
    else:
        pf = pd.DataFrame()
    
    return pf

#_____________________________________

# dfrq.py
def dfrq(ib, df_chains, exchange):
    '''Get remaining quantities
    Args:
        (ib) as connection object
        (df_chains) as chains dataframe with undPrice
        (exchange) as <'NSE'> | <'SMART'>
    Returns:
        dfrq as a dataframe of remaining quantities indexed on symbol'''

    # From Portfolio get the remaining quantitites
    p = util.df(ib.portfolio()) # portfolio table

    # extract option contract info from portfolio table
    if p is not None:  # there are some contracts in the portfolio
        dfp = pd.concat([p, util.df([c for c in p.contract])[util.df([c for c in p.contract]).columns[:7]]], axis=1).iloc[:, 1:]
        dfp = dfp.rename(columns={'lastTradeDateOrContractMonth': 'expiry'})

        # extract the options
        dfpo = dfp[dfp.secType == 'OPT']

        # get unique symbol, lot and underlying
        df_lu = df_chains[['symbol', 'lot', 'undId', 'undPrice']].groupby('symbol').first()

        # integrate the options with lot and underlying
        dfp1 = dfpo.set_index('symbol').join(df_lu).reset_index()

        # correct the positions for nse
        if exchange == 'NSE':
            dfp1 = dfp1.assign(position=dfp1.position/dfp1.lot)

        # get the total position for options
        dfp2 = dfp1[['symbol', 'position']].groupby('symbol').sum()

        # Get Stock positions
        dfs1 = p[p.contract.apply(lambda x: str(x)).str.contains('Stock')]
        if not dfs1.empty:
            dfs2 = util.df(list(dfs1.contract))
            dfs3 = pd.concat([dfs2.symbol, dfs1.position.reset_index(drop=True)], axis=1)
            dfs4 = dfs3.set_index('symbol').join(df_lu)
            dfs5 = dfs4.assign(position = (dfs4.position/dfs4.lot))[['position']]
            dfp2 = dfp2.add(dfs5, fill_value=0)  # Add stock positions to option positions

        # integrate position and lots and underlyings
        dfrq1 = df_lu.join(dfp2)

    else:
        print('There is nothing in the portfolio')
        dfrq1 = df_lu
        dfrq1['position'] = 0

    # fill in the other columns
    dfrq1 = dfrq1.assign(position=dfrq1.position.fillna(0)) # fillnas with zero
    dfrq1 = dfrq1.assign(assVal=dfrq1.position*dfrq1.lot*dfrq1.undPrice)

    assignment_limit = eval(market+'_assignment_limit')

    dfrq1 = dfrq1.assign(mgnQty=-(assignment_limit/dfrq1.lot/dfrq1.undPrice))
    dfrq1 = dfrq1.assign(remq=(dfrq1.position-dfrq1.mgnQty))
    dfrq = dfrq1.assign(remq=dfrq1.remq.fillna(0))[['remq']]

    dfrq.loc[dfrq.remq == np.inf, 'remq'] = 0  # remove them! They might be in the money.

    dfrq = dfrq.assign(remq=dfrq.remq.astype('int'))
    
    return dfrq

#_____________________________________

# StopExecution.py
class StopExecution(Exception):
    '''Stops execution in an iPython cell gracefully.
    To be used instead of exit()'''
    def _render_traceback_(self):
#         print(f'Gracefully exiting the cell :)')
        pass

#_____________________________________

# covers.py
def covers(ib, market, df_chains, df_ohlcsd, fspath):
    '''Generate covered calls and puts for assigned SNP options
    Works only for SNP, as NSE cannot hold assigned stocks
    Args:
        (ib) as connection object
        (market) as <'snp'|'nse'>
        (df_chains) as DataFrame from pd.read_pickle(fspath+'chains.pkl')
        (df_ohlcsd) as DataFrame from pd.read_pickle(fspath+'ohlcs.pkl')
        (fspath) as path of data file
    Returns: 
        df_covered as a DataFrame, ready for sells blocks and trade placement
        Also writes writecovers.pkl'''

    # abort program if market is NSE
    if market == 'nse':
        try:
            shell = get_ipython().__class__.__name__
        except NameError:
            exit() # Exit the interpreter
        raise StopExecution

    # get the portfolio
    pf = portf(ib)

    pf = pf.assign(shares=np.where(pf.secType == 'STK', pf.position, pf.position*100)) # get shares

    #...get the long and short stocks
    pfstk = pf[pf.secType == 'STK'] # stocks only

    # get the underlying price for stocks
    contracts = [Stock(symbol=s, exchange=exchange, currency=currency) for s in pfstk.symbol.unique()]
    qc = ib.qualifyContracts(*contracts)
    tickers = ib.reqTickers(*qc)
    undPrices = {t.contract.symbol: t.marketPrice() for t in tickers}

    pfstk = pfstk.assign(undPrice = [undPrices[p] for p in pfstk.symbol]) # get latest undPrices

    #...get the options for the stocks
    pfopts = pf[pf.secType == 'OPT'] # options only for the long and short stocks
    pfoptstk = pfopts.assign(undPrice = [catch(lambda: undPrices[p]) for p in pfopts.symbol]).dropna()

    pfstkopts = pd.concat([pfstk, pfoptstk]).sort_values(['symbol', 'secType'], ascending=[True, False])

    # ... make dataframe for covered calls
    # longs with covered calls
    lmask = (((pfstkopts.secType == 'STK') & (pfstkopts.shares > 0)) | ((pfstkopts.right == 'C') & (pfstkopts.shares < 0)))
    longs = pfstkopts[lmask]

    # get the sum of shares for the longs
    longshares = longs.groupby('symbol', as_index=False).agg({'shares': 'sum'})

    # remove those long symbols which has more call sells than what is available (exposed over cover) 
    longshares = longshares[longshares.shares > 0]

    # remove symbols in longshares that do not have underlying long stock
    longsymbols = pfstkopts[(pfstkopts.secType == 'STK') & (pfstkopts.shares > 0)].symbol
    longshares = longshares[longshares.symbol.isin(longsymbols)]

    # make it to a dictionary
    forcovcalls = longshares[['symbol', 'shares']]
    covcalls = dict(zip(forcovcalls.symbol, forcovcalls.shares))

    # ...make dataframe for covered puts
    # shorts with covered puts
    pmask = ((pfstkopts.secType == 'STK') & (pfstkopts.shares < 0)) | ((pfstkopts.right == 'P') & (pfstkopts.shares < 0))
    shorts = pfstkopts[pmask]

    # change the sign of short shares
    shorts = shorts.assign(shares = np.where(shorts.secType == 'STK', abs(shorts.shares), shorts.shares))

    # get the sum of shares for the shorts
    shortshares = shorts.groupby('symbol', as_index=False).agg({'shares': 'sum'})

    # remove symbols in shortshares that do not have underlying short stock
    shortsymbols = pfstkopts[(pfstkopts.secType == 'STK') & (pfstkopts.shares < 0)].symbol
    shortshares = shortshares[shortshares.symbol.isin(shortsymbols)]

    # remove those short symbols which have more put sells than what is available (expeosed over cover)
    shortshares = shortshares[shortshares.shares > 0]

    # make it to a dictionary
    forcovputs = shortshares[['symbol', 'shares']]
    covputs = dict(zip(forcovputs.symbol, abs(forcovputs.shares))) # puts show negative shares!

    # get the option chains for symbols needing covers
    needcovers = set(covcalls.keys()).union(set(covputs.keys()))
    df_chains = df_chains[df_chains.symbol.isin(needcovers)].reset_index(drop=True)

    # replace dte with 1 for dte <= 0
    df_chains.loc[df_chains.dte <=0,  'dte'] = 1
    df1 = df_chains[df_chains.dte <= maxdte]

    # assign right
    df1 = df1.assign(right=np.where(df1.strike >= df1.undPrice, 'C', 'P'))

    # generate std dataframe
    dfo = df_ohlcsd[['symbol', 'stDev']]  # lookup dataframe
    dfo = dfo.assign(dte=dfo.groupby('symbol').cumcount()) # get the cumulative count for location as dte
    dfo.set_index(['symbol', 'dte'])

    dfd = df1[['symbol', 'dte']]  # data to be looked at
    dfd = dfd.drop_duplicates()  # remove duplicates

    df_std = dfd.set_index(['symbol', 'dte']).join(dfo.set_index(['symbol', 'dte']))

    # join to get std in chains
    df2 = df1.set_index(['symbol', 'dte']).join(df_std).reset_index()

    # get the strikes of interest 1SD away from undPrice
    df2 = df2.assign(strikeRef = np.where(df2.right == 'P', df2.undPrice - df2.stDev*coverSD, df2.undPrice + df2.stDev*coverSD))

    #...overwrite strikeRef with averageCost, as appropriate

    # get the averagecost of the stocks
    costdict = {d['symbol']: d['averageCost'] for d in pfstk[['symbol', 'averageCost']].to_dict(orient='records')}
    df2 = df2.assign(costRef=[catch(lambda: costdict[s]) for s in df2.symbol])

    # for Puts...
    df2 = df2.assign(strikeRef = np.where((df2.right == 'P') & (df2.costRef < df2.strikeRef), df2.costRef, df2.strikeRef))
    # for Calls...
    df2 = df2.assign(strikeRef = np.where((df2.right == 'C') & (df2.costRef > df2.strikeRef), df2.costRef, df2.strikeRef))

    # get the options closest to the strikeRef
    df3 = df2.groupby(['symbol', 'dte'], as_index=False) \
                     .apply(lambda g: g.iloc[abs(g.strike - g.strikeRef) \
                     .argsort()[:2]]) \
                     .reset_index(drop=True)

    # choose the minimum dte for the covers
    df4 = df3.loc[df3.groupby(['right', 'symbol']).dte.idxmin()]

    # get the target covered call dataframe
    mask = (df4.right == 'C') & (df4.symbol.isin(covcalls.keys())) | \
           (df4.right == 'P') & (df4.symbol.isin(covputs.keys()))
    df5 = df4[mask]

    # determine the quantities for covered calls/ puts
    mapper = {'C': covcalls,
              'P': covputs}

    df5 = df5.assign(qty=df5.groupby('right').symbol.apply(lambda s: s.map(mapper[s.name])/100))

    # remove unnecessary calls and puts (that don't have underlying STK)
    df_covered = df5[((df5.right == 'C') & df5.symbol.isin(covcalls.keys())) | \
                     ((df5.right == 'P') & df5.symbol.isin(covputs.keys()))].reset_index(drop=True)



   # change the date of covered expiry to the coming week if dte <= 1
    newExp = df_covered.expiry.apply(lambda d: util.formatIBDatetime(util.parseIBDatetime(str(d)) + datetime.timedelta(days=7))[:8])

    mask = df_covered.dte <= 1
    df_covered.loc[mask, 'expiry'] = newExp[mask]

    # get the option prices
    covopts = ib.qualifyContracts(*[Option(c.symbol, c.expiry, c.strike, c.right, exchange) for i, c in df_covered.iterrows()])

#     # asyncio coroutine
#     async def coro():
#         tasks = [ib.reqTickersAsync(s) for s in covopts]
#         return await asyncio.gather(*tasks)

#     covticks = [c for r in ib.run(coro()) for c in r]

    if covopts:
        covticks = ib.reqTickers(*covopts)
        df_covered1 = df_covered.assign(optId = [i.conId for i in covopts])

        df_covered2 = df_covered1.join(pd.DataFrame([(c.bid, c.ask, c.marketPrice()) for c in covticks], columns=['bid', 'ask', 'mktPrice']))
        df_covered3 = df_covered2.dropna()

        # expected price as max of minexpOptprice, mktPrice and 3rd quartile of bid-ask spread
        expPrice = np.maximum(
                        np.maximum([get_prec(p, prec) for p in (df_covered3.ask-df_covered3.mktPrice)/2 + df_covered3.mktPrice], df_covered3.mktPrice + (prec*upthecoverfactor)), 
                        minexpOptPrice)

        df_covered4 = df_covered3.assign(expPrice = expPrice)
        
    else:
        df_covered4 = pd.DataFrame([])
    
    df_covered4.to_pickle(fspath+'writecovers.pkl')
    
    return df_covered4

#_____________________________________

# create_templates.py
def create_templates(fspath):
    '''Creates empty templates in bin/templates/ folder
    Arg: (fspath) - path from json
    Returns: Empty df_trades.pkl'''
    
    df = pd.read_pickle(fspath+'targets.pkl')
    df.drop(df.index, inplace=True)
    df.to_pickle('./templates/df_trades.pkl')
    
    print("\nEmpty df_trades template pickled\n")

#_____________________________________

# get_acc_summary.py
def get_acc_summary(ib):
    '''Gets a dictionary of account summary
    Arg: (ib) as connection object
    Returns: {dict} of account summary'''
    
    df_ac = util.df(ib.accountSummary())
    NLV = float(df_ac[df_ac.tag.isin(['NetLiquidation'])].value.iloc[0])
    initMargin = float(df_ac[df_ac.tag.isin(['InitMarginReq'])].value.iloc[0])
    unrealPnL = float(df_ac[df_ac.tag.isin(['UnrealizedPnL'])].value.iloc[0])
    realPnL = float(df_ac[df_ac.tag.isin(['RealizedPnL'])].value.iloc[0])
    avFunds = float(df_ac[df_ac.tag.isin(['AvailableFunds'])].value.iloc[0])
    acsum = {"NLV": NLV, "initmargin": initMargin, "unrealzPnL": unrealPnL, 
             "realzPnL": realPnL, "avFunds": avFunds}
    
    return acsum    

#_____________________________________

