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
    '''Pickles snp option chains
    Arg: (ib) as connection object
     Returns: None. But pickles to opts.pickle'''
    
    # exclusion list (excludes symbols with existing positions!)
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

#     #######  DATA LIMITER   ########
#     positions = ib.positions()
#     symbols = list({p.contract.symbol for p in positions})[:3]
#     ################################

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

    # mega dataframe - and pickle
    df_mega = pd.merge(df_clmu, df_es)
    
    df_mega.to_pickle(fspath+'base.pickle')
    util.logging.info('_________________SNP BASE BUILD COMPLETE_________________________')

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

# remqty_snp.py
def remqty_snp(ib):
    '''generates the remaining quantities dictionary
    Args:
        (ib) as connection object
        <df_opt> as dataframe of options. If not provided defaults to opts.pickle
    Returns:
        dfrq as a datframe with symbol and remaining quantities
        '''
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

    # make remaining quantity 1 for high margin stocks
    df_opt.loc[df_opt.und_remq <=0, 'und_remq'] = 1

    # compute the remaining quantities
    df_opt1 = df_opt.groupby('symbol').first()[['lot', 'margin', 'und_remq']]

    df_opt2 = df_opt1.join(dfp1).fillna(0).astype('int')
    df_opt2 = df_opt2.assign(remqty=df_opt2.und_remq+(df_opt2.position).astype('int'))

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
    
    watchlists(ib) # creates the watchlist
    
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
    df = df.assign(expPrice=np.where(df.rom < minexpRom, 
                                       get_prec((df.optPrice*minexpRom/df.rom), prec), 
                                       df.optPrice+0.05))

    df.loc[df.expPrice < minOptPrice, 'expPrice'] = minOptPrice # set to minimum optPrice
    
    df = df.assign(expRom=df.expPrice*df.lot/df.optMargin*365/df.dte). \
          sort_values('rom', ascending=False) # calculate the expected rom
    
    df = df.dropna().reset_index(drop=True)
    
    #... Establish sowing quantities
    
    putFall = (df.right == 'P') & ((df.undPrice - df.strike) > df.Fall)
    callRise = (df.right == 'C') & ((df.strike - df.undPrice) > df.Rise)

    df = df.assign(qty=np.where(callRise | putFall, df.remqty, 1)) # make quantities to be 1 for risky options
    df = df.assign(qty=np.where(df.qty <= maxsellqty, df.qty, maxsellqty)) # limit sellquantities
    
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

    x = []
    def getTickers(tickers):
        x.append(tickers)
    ib.pendingTickersEvent += getTickers

    optPrices = {t.contract.conId: t.marketPrice() for t in tickers} # {symbol: undPrice}
    dfopt = dfopt.assign(optPrice = dfopt.optId.map(optPrices))

    # get the margins
    orders = [Order(action='SELL', orderType='MKT', totalQuantity=1, whatIf=True) for lot in dfopt.lot]

    mdict = {c.conId: catch(lambda: float(ib.whatIfOrder(c, o).initMarginChange)) for c, o in zip(contracts, orders)} # making dictionary takes time!

    # Remove very large number in margin (1.7976931348623157e+308)
    mdict = {key: value for key, value in mdict.items() if float(value) < 1e7}

    # assign the margins
    dfopt = dfopt.assign(optMargin=dfopt.optId.map(mdict))

    # calculate rom
    dfopt = dfopt.assign(rom=dfopt.optPrice*dfopt.lot/dfopt.optMargin*365/dfopt.dte)

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
    '''For dynamic (live) trading (~3 mins)
    Arg: (ib) as connection object
    Returns: None. Does the following:
      1) Places BUY trades for those that don't have closing (harvest) trades
      2) If overall margin limit is busted, cancels all SELL trades
      3) '''

    sell_blks = [] # Initialize sell blocks

    # ... make the input dataframes
    #______________________________

    df_ac = util.df(ib.accountSummary()) # account summary

    # ..positions

    df_p = util.df(ib.positions())

    df_pc = util.df(list(df_p.contract))
    df_pc = df_pc[list(df_pc)[:6]]
    df_pc = df_pc.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})

    df_pc = df_pc.join(df_p[['position', 'avgCost']]) # merge contract with postion and avgCost

    df_pos = df_pc.assign(avgCost=np.where(df_pc.secType == 'OPT', df_pc.avgCost/100, df_pc.avgCost))

    # ..fill symbols
    if ib.fills():
        df_fc = util.df(list(util.df(ib.fills())['contract'])) # filled contracts
        df_fc = df_fc[list(df_fc)[:6]]
        df_fc = df_fc.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})
        df_fc

        df_fe = util.df(list(util.df(ib.fills())['execution'])) # executed contracts
        df_fe = df_fe[list(df_fe)[4:13]].drop('liquidation', 1) 

        df_fill = df_fc.join(df_fe)

        fill_symbols = list(df_fill.symbol)

    else:
        fill_symbols = []

    # ..open orders
    open_trades = ib.openTrades()

    if open_trades:
        df_open=util.df(open_trades)

        # open contracts
        df_oc = util.df(list(df_open['contract']))
        df_oc = df_oc[list(df_oc)[:6]]
        df_oc = df_oc.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})

        # orders
        df_ord = pd.DataFrame([(row.orderId, row.permId, row.action, row.totalQuantity, row.lmtPrice) 
                  for idx, row in df_open.order.items()],
                columns = ['orderId', 'permId', 'action', 'totalQuantity', 'lmtPrice'])

        df_openord = df_oc.join(df_ord)

        # filter the existing sells
        df_sx = df_openord[df_openord.action == 'SELL'] 

    else:
        df_openord = pd.DataFrame() # Empty open order DataFrame
        df_sx = pd.DataFrame() # Empty SELL open order DataFrame

    df_t = targets(ib) # get the targets from opts.pickle, sanitized for remqty

    df_buy = df_pos[df_pos.secType == 'OPT'] # filter out any assigned stock

    # ... prepare the BUY harvest orders
    #_________________________________

    if not df_openord.empty:
        df_bx = df_openord[df_openord.action == 'BUY'] # existing buys
        df_buy = df_buy[~df_buy.conId.isin(list(df_bx.conId))]    

    # ..Get the buy option prices
    buy_opts = ib.qualifyContracts(*[Contract(conId=c) for c in df_buy.conId])
    buy_ticks = ib.reqTickers(*buy_opts)
    buyOptPrices = {t.contract.conId: t.marketPrice() for t in buy_ticks} # {symbol: undPrice}

    df_buy = df_buy.assign(optPrice = df_buy.conId.map(buyOptPrices))
    df_buy = df_buy.assign(optPrice = df_buy.optPrice.fillna(prec)) # make NaN to lowest price (prec)

    #..get harvest price

    # dte
    df_buy = df_buy.assign(dte=[get_dte(d) for d in df_buy.expiration])

    # first hvstPrice
    df_buy = df_buy.assign(hvstPrice=[get_prec(min(hvstPricePct(d)*c,mp*0.9),prec) 
                                  for d, c, mp in zip(df_buy.dte, df_buy.avgCost, df_buy.optPrice)])

    # adjust to lower hvstPrice for hvstPrice = optPrice
    df_buy = df_buy.assign(hvstPrice=np.where((df_buy.optPrice == df_buy.hvstPrice) &
                 (df_buy.hvstPrice > prec), df_buy.optPrice - prec, df_buy.hvstPrice))

    #...prepare to place the orders
    buy_orders = [LimitOrder(action='BUY', totalQuantity=-position, lmtPrice=hvstPrice) 
                          for position, hvstPrice in zip(df_buy.position, df_buy.hvstPrice)]

    #...prepare to place the orders
    buy_orders = [LimitOrder(action='BUY', totalQuantity=-position, lmtPrice=hvstPrice) 
                          for position, hvstPrice in zip(df_buy.position, df_buy.hvstPrice)]

    hco = list(zip(buy_opts, buy_orders))

    # ... Prepare the SELL sow orders
    #________________________________

    # .. find out margin breach
    margin_breached = False
    net_liq = float(df_ac[df_ac.tag == 'NetLiquidation'][:1].value)
    init_margin = float(df_ac[df_ac.tag == 'InitMarginReq'][:1].value)

    if init_margin >= net_liq*ovallmarginlmt:  # if overall limit is breached
        print("Overall margin limit breached")
        margin_breached = True

    if margin_breached == True:

        # cancel all SELL orders
        if not df_openord.empty:
            sell_cancel = [o for x in df_sx.orderId 
               for o in ib.openOrders() 
               if x == o.orderId]

            cancels = [ib.cancelOrder(c) for c in sell_cancel]

    else: # overall limit is not breached

        if fill_symbols: # if there are fill symbols

            if not df_sx.empty: # if there are existing SELL orders

                # cancel existing SELLS
                dfsx_cancel = df_sx[df_sx.symbol.isin(fill_symbols)]
                sell_cancel = [o for x in dfsx_cancel.orderId 
                   for o in ib.openOrders() 
                   if x == o.orderId]

                cancels = [ib.cancelOrder(c) for c in sell_cancel]

            # make target of the cancelled SELLS
            df_sell = df_t[df_t.symbol.isin(fill_symbols)]

        else: # there are no fill symbols

            if df_sx.empty:      # if existing SELL orders are not available
                df_sell = df_t   # all df_targets need to be sold

            else:        # if there openorders exists but is missing some targets
                if not df_openord.empty:
                    df_sell = df_t[~df_t.optId.isin(list(df_openord.conId))] # get the targets
                else:
                    df_sell = pd.DataFrame([]) # empty dataframe

        # refresh the prices for df_sell
        if not df_sell.empty:
            opts = ib.qualifyContracts(*[Contract(conId=c) for c in df_sell.optId])
            tickers = ib.reqTickers(*opts)
            optPrices = {t.contract.conId: t.marketPrice() for t in tickers} # {symbol: undPrice}
            df_sell = df_sell.assign(optPrice=df_sell.optId.map(optPrices))

            # adjust expPrice to be the max of expPrice and riskyPrice
            df_sell = df_sell.assign(expPrice=pd.DataFrame([df_sell.optId.map(riskyprice(df_sell, prec)), 
                                             df_sell.expPrice]).max())

            # recalculate expRom
            df_sell.assign(expRom=df_sell.expPrice*df_sell.lot/df_sell.optMargin*365/df_sell.dte)

            # prepare the sell blocks
            sell_blks = trade_blocks(ib, df_sell, exchange=exchange)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # WARNING: THE FOLLOWING CODE PLACES TRADES
    # _________________________________________
    if hco:
        hvstTrades = [ib.placeOrder(c, o) for c, o in hco]
    if sell_blks:
        sowTrades = doTrades(ib, sell_blks)

    return None

#_____________________________________

