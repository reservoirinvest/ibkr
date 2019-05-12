# imports.py
import pandas as pd
import requests
from io import StringIO
from itertools import product, repeat
from os import listdir
import logging
from bs4 import BeautifulSoup
import csv
from collections import defaultdict

from ib_insync import *

from helper import *

# do the assignments from JSON
a = assign_var('snp')
for v in a:
    exec(v)

#_____________________________________

# opts.py
def opts(ib):
    '''Pickles snps underlying (1 minute)
    Arg: (ib) as connection object
     Returns: None. But pickles to opts.pickle'''
    
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

#     #######    DATA LIMITER !!!!    #######
#     df_clmu = df_clmu[:3]
#     qcs_dict = {k: qcs_dict[k] for k in list(qcs_dict)[:3]}
#     #######################################

    # get undId
    df_clmu = df_clmu.assign(undId=[c.conId for c in df_clmu.contract])

    # reset index
    df_clmu = df_clmu.rename_axis('symbol').reset_index()

    # get the chains
    qundc = [v for k, v in qcs_dict.items()]

    rawchains = {c: ib.reqSecDefOptParams(underlyingSymbol=c.symbol, 
                                          underlyingSecType='STK', 
                                          underlyingConId=c.conId, 
                                          futFopExchange='') 
                 for c in qundc}

    chains = {k.symbol: j for k, v in rawchains.items() for j in v if j.exchange == exchange}

    # integrate the strikes and expiries
    exps = {k: v.expirations for k, v in chains.items()}
    strikes = {k: v.strikes for k, v in chains.items()}

    es = [list(product([k1], v1, v2)) for k1, v1 in exps.items() for k2, v2 in strikes.items() if k1 == k2]

    df_es = pd.DataFrame([s for r in es for s in r], columns=['symbol', 'expiration', 'strike'])

    # make the dataframe to be in between min and max dte
    df_es = df_es.assign(dte=[get_dte(d) for d in df_es.expiration])
    df_es = df_es[df_es.dte.between(mindte, maxdte)].reset_index(drop=True)

    # mega dataframe
    df_mega = pd.merge(df_clmu, df_es)

    # get the max fall and rise

    df_frs = df_mega.drop('strike', 1).drop_duplicates() # for fall rise std
    mfrs = [((s, d) + get_maxfallrisestd(ib=ib, c=c, dte=d, tradingdays=tradingdays, durmult=durmult)) for s, c, d in zip(df_frs.symbol, df_frs.contract, df_frs.dte)]

    df_mfrs = pd.DataFrame(mfrs, columns=['symbol', 'dte', 'lo52', 'hi52', 'Fall', 'Rise', 'Std'])

    df_mega = df_mega.set_index(['symbol', 'dte']).join(df_mfrs.set_index(['symbol', 'dte'])).reset_index()
    df_mega['loStd'] = df_mega.undPrice - df_mega['Std'].multiply(putstdmult)
    df_mega['hiStd'] = df_mega.undPrice + df_mega['Std'].multiply(callstdmult)

    # flter puts and calls by standard deviation
    df_puts = df_mega[df_mega.strike < df_mega.loStd]
    df_calls = df_mega[df_mega.strike > df_mega.hiStd]

    df_puts = df_puts.assign(right='P')
    df_calls = df_calls.assign(right='C')

    # qualify the options
    df = pd.concat([df_puts, df_calls]).reset_index(drop=True).drop('contract', axis=1)

    # qualify the option contracts
    optipl = [Option(s, e, k, r, exchange) for s, e, k, r in zip(df.symbol, df.expiration, df.strike, df.right)]
    optblks = [optipl[i: i+blk] for i in range(0, len(optipl), blk)] # blocks of optipl
    cblks = [ib.qualifyContracts(*s) for s in optblks]
    contracts = [z for x in cblks for z in x]

    # prepare the qualified options dataframe
    dfq = util.df(contracts).iloc[:, 1:6]
    dfq.columns=['optId', 'symbol', 'expiration', 'strike', 'right'] # rename columns
    dfq = dfq.set_index(['symbol', 'expiration', 'strike', 'right']) # set index

    # join the qualified options with main dataframe. Remove those without conId
    df_opt = df.set_index(['symbol', 'expiration', 'strike', 'right']).join(dfq).dropna()

    # convert optId to int for price and margin
    df_opt = df_opt.assign(optId=df_opt.optId.astype('int32'))

    # get the prices
    opt_tickers = ib.reqTickers(*contracts)
    optPrices = {t.contract.conId: t.marketPrice() for t in opt_tickers}

    df_opt = df_opt.assign(optPrice=df_opt.optId.map(optPrices))
    df_opt = df_opt[~(df_opt.optPrice <= 0)].reset_index() # remove options without a price

    #... get the margins
    # {conId: contract} dictionary
    c_dict = {c.conId: c for c in contracts if c.conId in list(df_opt.optId)}
    orders = [Order(action='SELL', orderType='MKT', totalQuantity=1, whatIf=True)]*len(c_dict)

    # get the margins
    margins = [ib.whatIfOrder(c, o).initMarginChange for c, o in zip(c_dict.values(), orders)]

    df_opt = df_opt.assign(optMargin = margins)
    df_opt = df_opt.assign(optMargin = abs(pd.to_numeric(df_opt.optMargin).astype('float'))) # convert to float
    df_opt = df_opt[df_opt.optMargin < 1e7] # remove options with very large margins

    # get the rom
    df_opt = df_opt.assign(rom=df_opt.optPrice*df_opt.lot/df_opt.optMargin*365/df_opt.dte).sort_values('rom', ascending=False)

    df_opt.to_pickle(fspath+'opts.pickle')
    
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
        dfrq as a datframe with symbol and remaining quantities
        '''    # from portfolio
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
    df_opt = df_opt.assign(und_remq=(snp_assignment_limit/(df_opt.lot*df_opt.undPrice)).astype('int')) # remaining quantities in entire snp

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
    dfrq = remqty_snp(ib) # remaining quantities df

    df1 = pd.read_pickle(fspath+'opts.pickle')

    # filter for high probability and margin
    df2 = df1[((df1.strike > df1.hi52) | 
               (df1.strike < df1.lo52))].sort_values('rom', ascending=False)

    df2 = df2[df2.margin < 1e+308]  # Remove very large number in margin (1.7976931348623157e+308)

    df2 = df2.assign(remqty=df2.symbol.map(dfrq.remqty.to_dict()))   # remaining quantities
    df2 = df2[df2.remqty > 0].reset_index(drop=True) # remove blacklisted (remqty <= 0)

    df2 = df2.groupby('symbol').head(nLargest) # get the top 3

    # generate expPrice
    df3 = gen_expPrice(ib, df2)
    
    df3.to_pickle(fspath+'targets.pickle') # pickle the targets
    
    watchlist(ib) # creates the watchlist
    
    return df3

#_____________________________________

# gen_expPrice.py
def gen_expPrice(ib, df):
    '''generates expected price
    Args: 
        (ib) as the connection object
        (df) as the target options from _targetopt.pickle
    Returns:
        updated df with new recalculated expPrice and expRom
    '''
    df = df.assign(expPrice=np.where(df.rom < minRom, 
                                       get_prec((df.optPrice*minRom/df.rom), 0.05), 
                                       df.optPrice+0.05))

    df.loc[df.expPrice < minOptPrice, 'expPrice'] = minOptPrice # set to minimum optPrice
    
    df = df.assign(expRom=df.expPrice*df.lot/df.margin*365/df.dte). \
          sort_values('rom', ascending=False) # calculate the expected rom
    
    df = df.dropna().reset_index(drop=True)
    
    #... Establish sowing quantities
    
    putFall = (df.right == 'P') & ((df.undPrice - df.strike) > df.Fall)
    callRise = (df.right == 'C') & ((df.strike - df.undPrice) > df.Rise)

    df = df.assign(qty=np.where(callRise | putFall, df.remqty, 1))
    
    df = grp_opts(df)

    return df

#_____________________________________

# upd.py
def upd(ib, dfopt):
    '''Updates the snp options' price and margin
    Takes 3 mins for 450 options
    Args:
       (ib) as connection object
       (dfopt) as the dataframe from targets.pickle
    Returns: DataFrame with updated undPrice and margin'''
    
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
    
    return df

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
    watch = [('DES', s, 'STK', 'SMART') for s in sy]

    # write to watch.csv
    util.df(watch).to_csv(fspath+'watch.csv', header=None, index=None)
    
    # write targets to csv
    df.to_csv(fspath+'targets.csv', index=None, header=True)

#_____________________________________

# dynamic.py
def dynamic(ib):
    '''For dynamic (live) trading
    Arg: (ib) as connection object
    Returns: None. Does the following:
      1) Cancels openorders of filled symbols to reduce risk
      2) Finds positions not having closing (harvest) orders. Places them.
      3) Recalculates remaining quantities with a 10% higher price
      4) Sows them
      5) Re-pickles the target'''
    
    # ...get filled dataframe
    df_fill = util.df(list(util.df(ib.fills())['contract']))
    df_fill = df_fill[list(df_fill)[:6]]
    df_fill = df_fill.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})

    # ...get openorder dataframe
    df_open = util.df(ib.openTrades())

    # open contracts
    df_oc = util.df(list(df_open['contract']))
    df_oc = df_oc[list(df_oc)[:6]]
    df_oc = df_oc.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})

    # orders
    df_ord = util.df(list(df_open['order']))
    df_ord = df_ord[list(df_ord)[1:8]]

    df_openord = df_oc.join(df_ord).drop('clientId', axis=1)

    # ... cancel SELL openorders for those symbols that have been filled
    # get the symbols of orders filled
    fill_symbols = list(df_fill.symbol.unique())

    # get the openorders that have fill symbols (to cancel and harvest)
    ord_mask = df_openord.symbol.isin(fill_symbols) & (df_openord.action == 'SELL')
    df_hvst = df_openord[ord_mask] # harvest orders

    # cancel the openorders
    cancel_list = [o for o in ib.openOrders() if o.orderId in list(df_hvst.orderId)]
    cancelled = [ib.cancelOrder(c) for c in cancel_list]

    # ... find postions not having closing / harvesting (BUY) orders

    # get the positions
    df_p = util.df(ib.positions())

    # get the position contracts
    df_pc = util.df(list(df_p.contract))
    df_pc = df_pc[list(df_pc)[:6]]
    df_pc = df_pc.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})

    df_pc = df_pc.join(df_p[['position', 'avgCost']])

    # get the existing BUY orders
    df_bx = df_openord[df_openord.action == 'BUY']

    # determine what to harvest
    df_buy = df_pc[~df_pc.conId.isin(list(df_bx.conId))]

    df_buy = df_buy.assign(dte=[get_dte(d) for d in df_buy.expiration])

    # get the latest prices
    contracts = ib.qualifyContracts(*[Contract(conId=c) for c in df_buy.conId])

    tickers = ib.reqTickers(*contracts)
    optPrices = {t.contract.conId: t.marketPrice() for t in tickers} # {conId: optPrice}

    df_buy = df_buy.assign(optPrice = df_buy.conId.map(optPrices))

    # get the harvest price
    df_buy = df_buy.assign(hvstPrice=[get_prec(min(hvstPricePct(d)*c,mp*0.9),0.05) for d, c, mp in zip(df_buy.dte, df_buy.avgCost, df_buy.optPrice)])

    buy_orders = [LimitOrder(action='BUY', totalQuantity=-position, lmtPrice=hvstPrice) 
                              for position, hvstPrice in zip(df_buy.position, df_buy.hvstPrice)]

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # WARNING: THE FOLLOWING CODE PLACES 'HARVEST' ORDERS
    # ___________________________________________________
    hco = zip(contracts, buy_orders)
    hvstTrades = [ib.placeOrder(c, o) for c, o in hco]
    # @@@@@@@@    END OF ORDER PLACEMENT   @@@@@@@@@@@@@@

    # get the remaining quantities
    dfrq = remqty_snp(ib)

    # update target with remaining quantities
    dft = pd.read_pickle(fspath+'targets.pickle')

    dft = dft.assign(remqty=dft.symbol.map(dfrq.remqty.to_dict()))   # remaining quantities
    dft = dft[dft.remqty > 0].reset_index(drop=True) # remove blacklisted (remqty <= 0)

    # ...recalculate the price and margins for position symbols in dft
    df_rcalc = dft[dft.symbol.isin(fill_symbols)]

    # if rcalc is not empty recalculate, sow and write-back to targets pickle
    if not df_rcalc.empty:

        df_rcalc = upd(ib, df_rcalc)
        df_rcalc = df_rcalc.assign(expPrice=get_prec(df_rcalc.expPrice*1.1,0.05))

        # place new sow trades from df_rcalc
        sell_orders = [LimitOrder(action='SELL', totalQuantity=q*l, lmtPrice=expPrice) for q, l, expPrice in zip(df_rcalc.qty, df_rcalc.lot, df_rcalc.expPrice)]
        # get the contracts
        cs=[Contract(conId=c) for c in df_rcalc.optId]

        blks = [cs[i: i+blk] for i in range(0, len(cs), blk)]
        cblks = [ib.qualifyContracts(*s) for s in blks]
        qc = [z for x in cblks for z in x]

        co = list(zip(qc, sell_orders))
        coblks = [co[i: i+blk] for i in range(0, len(co), blk)]

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # WARNING: THE FOLLOWING CODE PLACES 'SOW' ORDERS
        # _______________________________________________
        trades = []
        for coblk in coblks:
            for co in coblk:
                trades.append(ib.placeOrder(co[0], co[1]))
            ib.sleep(1)
        # @@@@@@@@    END OF ORDER PLACEMENT   @@@@@@@@@@@@@@

        # remove old df_rcalc from dft
        dft = dft[~dft.optId.isin(list(df_rcalc.optId))]

        # append new df_rcalc to dft
        dft.append(df_rcalc).reset_index(drop=True)

        # write back to targets pickle
        dft.to_pickle(fspath+'targets.pickle')
        
        return None

#_____________________________________

