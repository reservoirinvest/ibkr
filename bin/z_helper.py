# imports_headers.py

"""Common helper programs
Date: 23-July-2019
Ver: 1.0
Time taken: milliseconds to load
"""

import pandas as pd
import numpy as np
import requests
import calendar
import time
import datetime
import logging
import csv
import json
import sys

import asyncio

import matplotlib.pyplot as plt

from ib_insync import *

from io import StringIO
from itertools import product, repeat
from os import listdir, path, unlink
from bs4 import BeautifulSoup
from tqdm import tqdm, tnrange

from math import floor, log10, ceil
from lxml import html

# for requests
headers = { 
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36', 
'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 
'Accept-Language' : 'en-US,en;q=0.5',
'Accept-Encoding' : 'gzip', 
'DNT' : '1', # Do Not Track Request Header 
'Connection' : 'close'
}


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

# from json
a = assign_var('common')
for v in a:
    exec(v)

#_____________________________________

# init_variables.py
def init_variables(market):
    '''Initializes variables from json
    Arg: (market) as string <'nse'> | <'snp'
    Outputs: None. But sets the varables'''
    
    # from json
    a = assign_var('common') + assign_var(market)
    for v in a:
        exec(v)
        
    return a

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

# do_hist.py
def do_hist(ib, undId, fspath):
    '''Historize ohlc
    Args:
        (ib) as connection object
        (undId) as contractId for underlying symbol in int
        (fspath) as string with pathname for the OHLCs
    Returns:
        df_hist as dataframe with running standard deviation in stDev
        pickles the dataframe by symbol name
    '''
    qc = ib.qualifyContracts(Contract(conId=int(undId)))[0]
    hist = ib.reqHistoricalData(contract=qc, endDateTime='', 
                                        durationStr='365 D', barSizeSetting='1 day',  
                                                    whatToShow='Trades', useRTH=True)
    df_hist = util.df(hist)
    df_hist = df_hist.assign(date=pd.to_datetime(df_hist.date, format='%Y-%m-%d'))
    df_hist.insert(loc=0, column='symbol', value=qc.symbol)
    df_hist = df_hist.sort_values('date', ascending = False).reset_index(drop=True)
    df_hist=df_hist.assign(stDev=df_hist.close.expanding(1).std(ddof=0))
    df_hist.to_pickle(fspath+'_'+qc.symbol+'_ohlc.pkl')
    return df_hist

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
        (df_hist) as a df with historical ohlc
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

# getMarginAsync.py
async def getMarginAsync(ib, c, o):
    '''computes the margin
    Args:
        (ib) as connection object
        (c) as a contract
        (o) as an order
    Returns:
        {m}: dictionary of localSymbol: margin as float'''

    try:
        aw = await asyncio.wait_for(ib.whatIfOrderAsync(c, o), timeout=2) # waits for 2 seconds

    except asyncio.TimeoutError: # fails the timeout
        return {c.conId:np.nan} # appends a null for failed timeout

    # success!
    return {c.conId:aw}

#_____________________________________

# grp_opts.py
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

# trade_blocks.py
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

# riskyprice.py
def riskyprice(dft, prec):
    '''adjusts expPrice to accomodate fall-rise risk
    Args:
        (dft) as the options dataframe
        (prec) precision needed as int
    Returns:
        (riskyprice) as dictionary'''
    pmask = (dft.right == 'P') & (dft.undPrice-dft.strike-dft.expPrice < dft.fall)
    df_prisky = pd.merge(dft[pmask][['symbol', 'optId', 'strike', 'right', 'dte', 'undPrice', 'optPrice', 'expPrice', 'fall', 'rise', 'expRom', 'qty']], 
             pd.DataFrame((dft[pmask].undPrice-dft[pmask].strike-dft[pmask].expPrice)), on=dft[pmask].index).drop('key_0', 1)

    cmask = (dft.right == 'C') & (dft.strike-dft.undPrice-dft.expPrice < dft.rise)
    df_crisky = pd.merge(dft[cmask][['symbol', 'optId', 'strike', 'right', 'dte', 'undPrice', 'optPrice', 'expPrice', 'fall', 'rise', 'expRom', 'qty']], 
             pd.DataFrame((dft[cmask].strike-dft[cmask].undPrice-dft[cmask].expPrice)), on=dft[cmask].index).drop('key_0', 1)

    df_risky = pd.concat([df_prisky, df_crisky]).reset_index(drop=True)

    df_risky = df_risky.rename(columns={0: 'FallRise'})

    df_risky = df_risky.assign(FRpct=abs(np.where(df_risky.right == 'P', (df_risky.FallRise - df_risky.fall)/df_risky.FallRise, (df_risky.FallRise - df_risky.rise)/df_risky.FallRise)))

    df_risky = df_risky.sort_values('FRpct', ascending=False)

    df_risky = df_risky.assign(expPrice = get_prec(df_risky.FRpct*df_risky.expPrice, prec))

    return df_risky.set_index('optId')['expPrice'].to_dict()

#_____________________________________

# closest_margin.py
def closest_margin(ib, df_opt, exchange):
    '''find the margin of the closest strike
    Args:
        (ib) as connection object
        (df_opt) as a single symbol option df with strikes & undPrice
        (exchange) as <'NSE'> | <'SNP'>
    Returns:
        (df_opt) with the closest margin field'''
    #... find margin for closest strike
    closest = df_opt.loc[abs(df_opt.strike-df_opt.undPrice.unique()).idxmin(), :].to_dict() # closest strike to undPrice

    x_opt = Option(symbol=closest['symbol'], lastTradeDateOrContractMonth=closest['expiration'], \
                   strike=closest['strike'], right=closest['right'], exchange=exchange)

    ocm = ib.qualifyContracts(x_opt) # opt contract for margin

    ocm_ord = Order(action='SELL', orderType='MKT', totalQuantity=closest['lot'], whatIf=True)
    margin = float(ib.whatIfOrder(*ocm, ocm_ord).initMarginChange)

    df_opt = df_opt.assign(undMargin=margin, xStrike=ocm[0].strike)
    
    return df_opt

#_____________________________________

# codetime.py
def codetime(seconds):
    '''get a printable hh:mm:ss time of elapsed program
    Arg: (seconds) as float
    Returns: hh:mm:ss as string'''
    
    m, s = divmod(seconds,60)
    h, m = divmod(m, 60)
    
    return '{:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s))

#_____________________________________

# portf.py
def portf(ib):
    '''gives (fast) portfolio sorted by unrealizedPNL
    Arg: (ib) as connection object
    Returns: pf as portfolio dataframe'''
    # get the portfolio
    pf = util.df(ib.portfolio()).drop('account', 1)
    pc = util.df(list(pf.contract)).iloc[:, :6]
    pf = pc.join(pf.drop('contract',1)).sort_values('unrealizedPNL', ascending=True)
    pf.rename({'lastTradeDateOrContractMonth': 'expiry'}, axis='columns', inplace=True)
    dtes = {p.expiry: get_dte(p.expiry) for p in pf.itertuples() if p.secType == 'OPT'}
    pf = pf.assign(dte=pf.expiry.map(dtes))
    
    return pf

#_____________________________________

# get_position.py
def get_position(ib, exchange, fspath, currency, prec):
    '''Gets the position dataframe
    Arg: 
        (ib) as connection object
        (exchange) as the string <'NSE'>|<'SMART'>
        (fspath) as the path of data in string
        (currency) as currency <'INR'>|<'USD'>
        (prec) as precision in float
    Returns: pos_df DataFrame
    Dependancies: chains.pkl'''
    
    # get the base portfolio
    pf = portf(ib)

    # get lots
    df_chains = pd.read_pickle(fspath+'chains'+'.pkl').set_index(['symbol', 'expiry', 'strike'])[['undId', 'lot']]
    pf = pf.set_index(['symbol', 'expiry', 'strike']).join(df_chains).reset_index()

    cs = [Contract(conId=c, exchange=exchange, currency=currency) for c in list(pf.conId)] 
    ticks = ib.reqTickers(*cs)
    # ib.sleep(4)

    df_bidask = pd.DataFrame([(t.contract.conId, t.bid, t.ask, t.marketPrice()) for t in ticks], 
                 columns=['conId', 'bid', 'ask', 'mktPrice']).set_index('conId')

    mod_dict = {t.contract.conId: t.modelGreeks for t in ticks if t.modelGreeks}

    if mod_dict:
        df_mod = pd.DataFrame(list(mod_dict.keys()), columns=['conId']).join(util.df(list(mod_dict.values()))).set_index('conId')
        pf = pf.set_index('conId').join(df_bidask.join(df_mod))
    else:
        pf = pf.set_index('conId').join(df_bidask)

    pf = pf.reset_index()

    # correct the averageCost for snp
    mask = pf.secType == 'OPT'
    if exchange == 'SMART':
        pf.loc[mask, 'averageCost'] = pf[mask].averageCost/pf[mask].lot

    # get the harvest price
    pf = pf.assign(hvstPrice=abs(pf.dte.map(hvstPricePct))*pf.averageCost)

    # rename averageCost to avgCost to differentiate between base portfolio (unadjusted by lots) and this
    pf.rename(columns={'averageCost': 'avgCost'}, inplace=True)

    # make expPrice
    if 'optPrice' in pf.columns:
        pf=pf.assign(expPrice=pf[['hvstPrice', 'marketPrice', 'mktPrice', 'optPrice', 'avgCost']].min(axis=1).apply(lambda x: get_prec(x, prec)))
    else:
        pf=pf.assign(expPrice=pf[['hvstPrice', 'mktPrice', 'avgCost']].min(axis=1).apply(lambda x: get_prec(x, prec)))

    #... get underlying price
    und_contracts = [Stock(symbol, exchange=exchange, currency=currency) for symbol in list(pf.symbol.unique())]
    und_quals = ib.qualifyContracts(*und_contracts)
    tickers = ib.reqTickers(*und_quals)

    uprice_dict = {u.contract.symbol: u.marketPrice() for u in tickers}

    pf=pf.assign(undPrice=pf.symbol.map(uprice_dict))

    pf_cols = ['conId', 'undId', 'secType', 'symbol', 'undPrice', 'right', 'strike', 'dte', 'expiry', 
               'position', 'avgCost', 'marketPrice', 'marketValue', 'unrealizedPNL', 'realizedPNL', 
               'lot', 'bid', 'ask', 'mktPrice', 'hvstPrice', 'expPrice']
    pf = pf[pf_cols].sort_values(['secType', 'unrealizedPNL']).reset_index(drop=True)
    
    return pf

#_____________________________________

# get_nearmoneyopt.py
def get_nearmoneyopt(pf):
    '''Gets risky options that are near the money / under loss
    Arg: (pf) as output of get_position(ib, exchange, fspath, currency, prec)
    Returns: dataframe of options sorted by risk '''
    
    # get the pnl_delta... which is strike-underlying + avgCost
    pf = pf.assign(pnlD=np.where(pf.right == 'P', pf.undPrice-pf.strike+pf.avgCost, pf.strike-pf.undPrice+pf.avgCost))

    # get the portfolios whose delta loss is greater than unrealized PnL
    df=pf[(pf.pnlD+pf.unrealizedPNL)<=0]

    # append the rows above first delta loss > unrealized PnL
    df=pd.concat([pf[:df[:1].index[0]], df])
    
    # keep only OPT
    df=df[df.secType == 'OPT']
    
    return df

#_____________________________________

# jup_disp_adjust.py
def jup_disp_adjust():
    '''Sets jupyter to show columns in 0.00 format and shows all columns'''
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.float_format', '{:.2f}'.format)

#_____________________________________

# cancel_sells.py
def cancel_sells(ib):
    '''Cancels all sell orders
    Arg: (ib) as connection object
    Returns: [canceld_sells] list'''
    # get all the trades
    trades = ib.trades()
    all_trades_df = util.df(t.contract for t in trades).join(util.df(t.orderStatus for t in trades)).join(util.df(t.order for t in trades), lsuffix='_')
    all_trades_df.rename({'lastTradeDateOrContractMonth': 'expiry'}, axis='columns', inplace=True)
    trades_cols = ['conId', 'symbol', 'localSymbol', 'secType', 'expiry', 'strike', 'right', 
                   'orderId', 'permId', 'action', 'totalQuantity', 'lmtPrice', 'status']
    trades_df = all_trades_df[trades_cols]

    # get the sell option trades which are open (SUBMITTED)
    df_open_sells = trades_df[(trades_df.action == 'SELL') & 
              (trades_df.secType == 'OPT') &
              (trades_df.status == 'Submitted')]

    # cancel the sell open orders
    sell_openords = [t.order for t in trades if t.order.orderId in list(df_open_sells.orderId)]
    canceld_sells = [ib.cancelOrder(order) for order in sell_openords]
    
    return canceld_sells

#_____________________________________

# place_morning_trades.py
def place_morning_trades(ib, buy_tb, sell_tb):
    '''Places morning trades
    Args:
        (ib) as connection object
        (buy_tb) as trade block list of buys from workout
        (sell_tb) as trade block list of sells from targets.pkl
    Returns:
        (buy_trades, sell_trades) executed tuple
    '''
    
    # get all the trades
    trades = ib.trades()

    if trades:
        all_trades_df = util.df(t.contract for t in trades).join(util.df(t.orderStatus for t in trades)).join(util.df(t.order for t in trades), lsuffix='_')
        all_trades_df.rename({'lastTradeDateOrContractMonth': 'expiry'}, axis='columns', inplace=True)
        trades_cols = ['conId', 'symbol', 'localSymbol', 'secType', 'expiry', 'strike', 'right', 
                       'orderId', 'permId', 'action', 'totalQuantity', 'lmtPrice', 'status']
        trades_df = all_trades_df[trades_cols]

        # get the sell option trades which are open (SUBMITTED)
        df_open_sells = trades_df[(trades_df.action == 'SELL') & 
                  (trades_df.secType == 'OPT') &
                  (trades_df.status == 'Submitted')]
    else:
        df_open_sells = pd.DataFrame()

    # clears all existing trades
    if not df_open_sells.empty:
        # cancel the sell open orders
        print("\nCancelling all open SELL trades...\n")
        sell_openords = [t.order for t in trades if t.order.orderId in list(df_open_sells.orderId)]
        canceld_sells = [ib.cancelOrder(order) for order in sell_openords]

    if buy_tb: # there is something to buy!
        print("\n Executing BUY trades ...")
        buy_trades = doTrades(ib, buy_tb)
        
    # check for margin breach
    ac_df=util.df(ib.accountSummary())[['tag', 'value']]

    if (float(ac_df[ac_df.tag == 'AvailableFunds'].value.values) /
        float(ac_df[ac_df.tag == 'NetLiquidation'].value.values)) < ovallmarginlmt:
        marginBreached = True

    # place sell trades if margin is not breached and there is something to sell
    if marginBreached:
        print("\nMARGIN BREACH!!! Will not execute SELLs")
        sell_trades = []
    elif sell_tb: # there is something to sell!
        print("\n Executing SELL Naked trades ...")
        sell_trades = doTrades(ib, sell_tb)
    else:
        sell_trades = []
        
    return (buy_trades, sell_trades)

#_____________________________________

# buy_sell_tradingblocks.py

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

