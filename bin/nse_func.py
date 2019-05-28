# imports.py
import pandas as pd
import requests
from io import StringIO
from itertools import product, repeat
from os import listdir
import logging
from bs4 import BeautifulSoup
import csv
from tqdm.auto import tqdm
import time
import asyncio

from ib_insync import *

from helper import *

# do the assignments from JSON
a = assign_var('nse')
for v in a:
    exec(v)

#_____________________________________

# symexplots.py
def symexplots(ib):
    '''Creates the symbol-expiry-lot.pickle with qualified underlyings
    Arg: (ib) as connection object
    Returns: 
        df_l as dataframe of lots
        Pickles df_l to symexplots.pickle'''

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

    #...get the expiry dates

    # make an expiry date dictionary
    res = requests.get('https://www.nseindia.com/live_market/dynaContent/live_watch/fomwatchsymbol.jsp?key=NIFTY&Fut_Opt=Futures')
    soup = BeautifulSoup(res.content, 'lxml')
    table = soup.find_all('table')
    df_exp = pd.read_html(str(table[1]), header=0)[0] # to dataframe
    dt = pd.to_datetime(df_exp['Expiry Date'])
    exp_dates = {d.strftime('%b-%y').lower(): d.strftime('%Y%m%d') for d in dt}

    # regenerate the columns using expiry date dictionary
    lots_df.columns = ['symbol'] + list(lots_df.set_index('symbol').columns.map(exp_dates))

    # remove symbols field
    lots_df = lots_df[~lots_df.symbol.isin(['Symbol'])]

    # force to appropriate int / float
    lots_df = lots_df.apply(pd.to_numeric, errors='ignore')

    df_l = pd.melt(lots_df.iloc[:, [0,1,2,3]], id_vars=['symbol'], var_name='expiration', value_name='lot').dropna()
    df_l = df_l.assign(ibSymbol=df_l.symbol.str.slice(0,9)).drop('symbol', axis=1) # for ibSymbol and drop symbol

    # make the lots into int64 for consistency
    df_l = df_l.assign(lot=df_l.lot.astype('int64'))

    # nseSymbol to ibSymbol dictionary for conversion
    ntoi = {'M&M': 'MM', 'M&MFIN': 'MMFIN', 'L&TFH': 'LTFH', 'NIFTY': 'NIFTY50'}

    # remap ibSymbol, based on the dictionary rename it to symbol
    df_l.ibSymbol = df_l.ibSymbol.replace(ntoi)

    df_l = df_l.rename(columns={"ibSymbol": "symbol"})

    # ... get the conIds and undPrices

    # set the types for indexes as IND
    ix_symbols = ['NIFTY50', 'BANKNIFTY', 'NIFTYIT']
    df_l = df_l.assign(scripType=np.where(df_l.symbol.isin(ix_symbols), 'IND', 'STK'))

    # build the underlying contracts
    scrips = list(df_l.symbol.unique())
    und_contracts = [Index(symbol=s, exchange=exchange) if s in ix_symbols else Stock(symbol=s, exchange=exchange) for s in scrips]

    und_qc = ib.qualifyContracts(*und_contracts)

    # Get the prices and drop those without price
    undTickers = ib.reqTickers(*und_qc)
    undPrices = {t.contract.symbol: t.marketPrice() for t in undTickers} # {symbol: undPrice}

    # Get the underlying conIds
    undIds = {t.contract.symbol: t.contract.conId for t in undTickers} # {symbol: conId}
    df_l = df_l.assign(undId=df_l.symbol.map(undIds))

    df_l = df_l.assign(undPrice=df_l.symbol.map(undPrices)).dropna()

    # get the dte
    df_l = df_l.assign(dte=[get_dte(e) for e in df_l.expiration])

    # make the dataframe to be in between min and max dte (final list)
    df_l = df_l[df_l.dte.between(mindte, maxdte)].reset_index(drop=True)

    # arrange the columns:
    df_l = df_l[['symbol', 'scripType', 'undId', 'undPrice', 'dte', 'expiration', 'lot']]

    df_l.to_pickle(fspath+'symexplots.pickle')

    return df_l

#_____________________________________

# do_hist.py
def do_hist(ib, undId):
    '''Historize ohlc
    Args:
        (ib) as connection object
        (undId) as contractId for underlying symbol
    Returns:
        df_hist as dataframe
        pickles the dataframe by symbol name
    '''
    qc = ib.qualifyContracts(Contract(conId=undId))[0]
    hist = ib.reqHistoricalData(contract=qc, endDateTime='', 
                                        durationStr='365 D', barSizeSetting='1 day',  
                                                    whatToShow='Trades', useRTH=True)
    df_hist = util.df(hist)
    df_hist = df_hist.assign(date=pd.to_datetime(df_hist.date, format='%Y-%m-%d'))
    df_hist.insert(loc=0, column='symbol', value=qc.symbol)
    df_hist = df_hist.sort_values('date', ascending = False).reset_index(drop=True)
    df_hist.to_pickle(fspath+'_'+qc.symbol+'_ohlc.pickle')
    return df_hist

#_____________________________________

# do_an_opt.py
def do_an_opt(ib, row):
    '''Gets the options for a contract
    Args:
        (ib) as connection object
        (row_1) as data series row from df_l
    Returns:
        df_opt as options dataframe
        pickles df_opt to symbol.pickle
        '''
    chain = ib.reqSecDefOptParams(underlyingSymbol=row.symbol,
                                                underlyingSecType=row.scripType,
                                                underlyingConId=row.undId,
                                                futFopExchange='')

    strikes = [x for c in chain for x in c.strikes]

    expirations = [x for c in chain for x in c.expirations]

    rights = ['P', 'C']

    df_opt = pd.DataFrame(list(product([row.symbol], strikes, 
                                       expirations, rights, 
                                       [row.undPrice])))

    df_opt = df_opt.rename(columns={0:'symbol', 1:'strike', 
                                    2:'expiration', 3: 'right', 4: 'undPrice'})

    df_opt = df_opt.assign(lot=row.lot)

    df_opt = df_opt.assign(dte=[get_dte(d) for d in df_opt.expiration])

    # weed option contracts within standard deviations band
    df_hist = do_hist(ib, row.undId)

    df_hist = df_hist.assign(rollstd = df_hist.close.expanding(1).std())
    std4dte = {d: df_hist.rollstd[d] for d in pd.Series(df_opt.dte.unique())}

    df_opt = df_opt.assign(stdLimit=np.where(df_opt.right == 'P', df_opt.dte.map(std4dte)*putstdmult, df_opt.dte.map(std4dte)*callstdmult))
    df_opt = df_opt.assign(strkLimit = np.where(df_opt.right == 'P', df_opt.undPrice - df_opt.stdLimit, df_opt.undPrice + df_opt.stdLimit))

    x_mask = ((df_opt.right == 'P') & (df_opt.strike < df_opt.strkLimit)) | ((df_opt.right == 'C') & (df_opt.strike > df_opt.strkLimit))
    df_opt = df_opt[x_mask]

    # ... get the conId of the option contracts
    optipl = [Option(s, e, k, r, exchange) for s, e, k, r in zip(df_opt.symbol, df_opt.expiration, df_opt.strike, df_opt.right)]
    optblks = [optipl[i: i+blk] for i in range(0, len(optipl), blk)] # blocks of optipl

    # qualify the contracts
    cblks = [ib.qualifyContracts(*s) for s in optblks]
    contracts = [z for x in cblks for z in x]

    df_conId = util.df(contracts).iloc[:, 1:6]
    df_conId = df_conId.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})

    df_opt = pd.merge(df_opt, df_conId, how='left', on=['symbol', 'expiration', 'strike', 'right']).dropna()

    df_opt = df_opt.assign(optId = df_opt.conId.astype('int'))
    df_opt = df_opt.drop(['stdLimit', 'strkLimit', 'conId'], axis=1)

    # get the underlying margin from closest strike
    df_opt = closest_margin(ib, df_opt, exchange)

    #... get the margins

    # build contracts and orders
    opt_con_dict = {c.conId: c for c in contracts if c.conId in list(df_opt.optId)}
    opt_ord_dict = {i: Order(action='SELL', orderType='MKT', totalQuantity=lot) for i, lot in zip(df_opt.optId, df_opt.lot)}

    # combine contracts and orders
    opts = zip(df_opt.optId.map(opt_con_dict), df_opt.optId.map(opt_ord_dict))

    # gather and get the margins through asyncio, without the nan
    g = asyncio.gather(*[getMarginAsync(ib, c, o) for c, o in opts])
    opt_mgn_dict = {k: v for d in asyncio.run(g) for k, v in d.items()}

    df_opt = df_opt.assign(optMargin=df_opt.optId.map(opt_mgn_dict)).dropna()

     # ... get the prices
    opt_tickers = ib.reqTickers(*contracts)
    optPrices = {t.contract.conId: t.marketPrice() for t in opt_tickers}

    df_opt = df_opt.assign(optPrice=df_opt.optId.map(optPrices))
    df_opt = df_opt.assign(rom=df_opt.optPrice/df_opt.optMargin*365/df_opt.dte*df_opt.lot).sort_values('rom', ascending=False).reset_index(drop=True)

    # get the 52 week highs and lows
    df_opt = df_opt.assign(lo52 = df_hist.iloc[:tradingdays].low.min(), hi52 = df_hist.iloc[:tradingdays].high.max())

    df_opt.to_pickle(fspath+'_'+row.symbol+'_'+row.expiration+'.pickle')
    
    return df_opt

#_____________________________________

# get_opts.py
def get_opts(ib, df_l):
    '''get the options pickles
    Arg: 
        (ib) as connection object
        (df_l) as symbol-expiry-lot dataframe
    Returns:
        df_opts as the options list
        pickles df_opt for each symbol'''

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

    # parameters for ib.reqSecDefOptParams
    params = {q.symbol: (q.symbol, 'IND', q.conId) if q.symbol in indexes else (q.symbol, 'STK', q.conId) for q in qcs}

    # ... pickle the underlyings

    df_opts = pd.DataFrame([]) # initialize

    tq_qcs = tqdm(qcs)
    for qc in tq_qcs:
        tq_qcs.set_description("Processing %s" % qc.symbol) # tqdm output

        df_opt = do_an_opt(ib, qc, params, df_l, undPrices) # generates options list and pickles them

        # appends and pickles the options
        df_opts = df_opts.append(df_opt)
        df_opts.to_pickle(fspath+'opts.pickle')

        df_opt = pd.DataFrame([]) # initialize the dataframe
        
    return df_opts

#_____________________________________

# remqty_nse.py
def remqty_nse(ib):
    '''generates the remaining quantities dictionary
    Args:
        (ib) as connection object
        <df_opt> as dataframe of options. If not provided defaults to opts.pickle
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
    df_opt1 = df_opt.groupby('symbol').first()[['lot', 'optMargin', 'und_remq']]

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
                                       get_prec((df.optPrice*minexpRom/df.rom), 0.5), 
                                       df.optPrice+0.5))

    df.loc[df.expPrice < minOptPrice, 'expPrice'] = minOptPrice # set to minimum optPrice
    
    df = df.assign(expRom=df.expPrice*df.lot/df.margin*365/df.dte). \
          sort_values('rom', ascending=False) # calculate the expected rom
    
    df = df.dropna().reset_index(drop=True)
    
    #... Establish sowing quantities
    
    putFall = (df.right == 'P') & ((df.undPrice - df.strike) > df.Fall)
    callRise = (df.right == 'C') & ((df.strike - df.undPrice) > df.Rise)

    df = df.assign(qty=np.where(callRise | putFall, df.remqty, 1))
    df = df.assign(qty=np.where(df.qty <= maxsellqty, df.qty, maxsellqty)) # limit sellquantities
    
    df = grp_opts(df)

    return df

#_____________________________________

# upd.py
def upd(ib, dfopt):
    '''Updates the nse options' price and margin
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
    watch = [('DES', s, 'STK', 'NSE') if s not in ['NIFTY50', 'BANKNIFTY', 'NIFTYIT'] 
             else ('DES', s, 'IND', 'NSE') for s in sy]

    # write to watch.csv
    util.df(watch).to_csv(fspath+'watch.csv', header=None, index=None)
    
    # write targets to csv
    df.to_csv(fspath+'targets.csv', index=None, header=True)

#_____________________________________

# dynamic.py
def dynamic(ib):
    '''For dynamic (live) trading  (~3 mins)
    Arg: (ib) as connection object
    Returns: None. Does the following:
      1) Cancels openorders of filled symbols to reduce risk
      2) Finds positions not having closing (harvest) orders. Places them.
      3) Recalculates remaining quantities with a 10% higher price
      4) Sows them
      5) Re-pickles the target'''
    
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
    df_pos = df_pc.assign(avgCost=np.where(df_pc.secType == 'OPT', df_pc.avgCost, df_pc.avgCost))

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
            df_sell.assign(expRom=df_sell.expPrice*df_sell.lot/df_sell.margin*365/df_sell.dte)

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

