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


    df_l = df_l.assign(ibSymbol=df_l.symbol.str.slice(0,9)).drop('symbol', axis=1) # for ibSymbol and drop symbol

    # make the lots into int64 for consistency
    df_l = df_l.assign(lot=df_l.lot.astype('int64'))

    # nseSymbol to ibSymbol dictionary for conversion
    ntoi = {'M&M': 'MM', 'M&MFIN': 'MMFIN', 'L&TFH': 'LTFH', 'NIFTY': 'NIFTY50'}

    # remap ibSymbol, based on the dictionary
    df_l.ibSymbol = df_l.ibSymbol.replace(ntoi)

    # ... index option chains

    # get the option chain
    ix_symbols = ['NIFTY50', 'BANKNIFTY', 'NIFTYIT']
    ix_contracts = ib.qualifyContracts(*[Index(s, exchange) for s in ix_symbols])
    ix_optip = {c: ib.reqSecDefOptParams(underlyingSymbol=c.symbol, underlyingSecType='IND', underlyingConId=c.conId, futFopExchange='') for c in ix_contracts}

    # extract expiration
    exp_ind = {k.symbol: v[0].expirations for k, v in ix_optip.items()}

    # put into dataframe
    df_ix = pd.DataFrame([k for j in [list(product([k], v)) 
                              for k, v in exp_ind.items()] 
                                  for k in j], columns = ['ibSymbol', 'expiration'])

    # get the lots for indexes
    df_ixlot = df_l[df_l.ibSymbol.isin(ix_symbols)][['ibSymbol', 'lot']].\
                    drop_duplicates().\
                    set_index('ibSymbol').\
                    to_dict()['lot']

    df_ix = df_ix.assign(lot=df_ix.ibSymbol.map(df_ixlot))

    # remove unnecessary symbols
    discards = ['LUPIN'] + ix_symbols # index symbols will be concatenated back
    df_l = df_l[~df_l.ibSymbol.isin(discards)]

    # concatenate index symbols
    df_l = pd.concat([df_ix, df_l], sort=False)

    # get the dte
    df_l = df_l.assign(dte=[get_dte(e) for e in df_l.expiration])

    # make the dataframe to be in between min and max dte (final list)
    df_l = df_l[df_l.dte.between(mindte, maxdte)].reset_index(drop=True)

    ######   DATA LIMITER #####
    # df_l = df_l.groupby('ibSymbol').first().loc[['ACC', 'BANKNIFTY'], :].reset_index()
    ###_____________________###
    
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

    # get max fall-rise std
    mfrs = [get_maxfallrisestd(ib=ib, c=c, dte=d, tradingdays=tradingdays, durmult=durmult) for c, d in zip(df_l.ibSymbol.map(qcs_dict), df_l.dte)]

    df_mfrs = pd.DataFrame(mfrs, columns=['lo52', 'hi52', 'Fall', 'Rise', 'Std'])
    df_lms = df_l.join(df_mfrs)

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
    '''For dynamic (live) trading
    Arg: (ib) as connection object
    Returns: None. Does the following:
      1) Cancels openorders of filled symbols to reduce risk
      2) Finds positions not having closing (harvest) orders. Places them.
      3) Recalculates remaining quantities with a 10% higher price
      4) Sows them
      5) Re-pickles the target'''
    
    #... get the open orders
    open_trades = ib.openTrades()

    if open_trades:
        df_open=util.df(open_trades)

        # open contracts
        df_oc = util.df(list(df_open['contract']))
        df_oc = df_oc[list(df_oc)[:6]]
        df_oc = df_oc.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})

        # orders
        df_ord = util.df(list(df_open['order']))
        df_ord = df_ord[list(df_ord)[1:8]]

        df_openord = df_oc.join(df_ord).drop('clientId', axis=1)

    else:
        df_openord = pd.DataFrame() # Empty DataFrame

    #... place closing orders for those without them

    # get the positions
    df_p = util.df(ib.positions())

    # get the position contracts
    df_pc = util.df(list(df_p.contract))
    df_pc = df_pc[list(df_pc)[:6]]
    df_pc = df_pc.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})

    df_pc = df_pc.join(df_p[['position', 'avgCost']])

    # get the existing BUY orders
    if not df_openord.empty:
        df_bx = df_openord[df_openord.action == 'BUY']
    else:
        df_bx = pd.DataFrame() # Empty dataframe

    # determine what to harvest
    if not df_bx.empty:
        df_buy = df_pc[~df_pc.conId.isin(list(df_bx.conId))]
    else:
        df_buy = df_pc

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

    hco = zip(contracts, buy_orders)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # WARNING: THE FOLLOWING CODE PLACES 'HARVEST' BUY ORDERS
    # _______________________________________________________
    hvstTrades = [ib.placeOrder(c, o) for c, o in hco]
    # @@@@@@@@    END OF ORDER PLACEMENT   @@@@@@@@@@@@@@

    #.... check for overall position risk. Abort after cancelling SELL orders at risk
    df_ac = util.df(ib.accountSummary())
    net_liq = float(df_ac[df_ac.tag == 'NetLiquidation'][:1].value)
    init_margin = float(df_ac[df_ac.tag == 'InitMarginReq'][:1].value)

    if init_margin >= net_liq*ovallmarginlmt:  # if overall limit is breached
        print("Overall margin limit breached")

        # ...cancel all SELL open orders, if it is not empty
        if not df_openord.empty:
            m_opord = (df_openord.action == 'SELL')
            df_sells = df_openord[m_openord]

            # cancel the SELL openorders
            cancel_list = [o for o in ib.openOrders() if o.orderId in list(df_sells.orderId)]
            cancelled = [ib.cancelOrder(c) for c in cancel_list]

            return None # remove the return function in dynamic live

    #... place SOW orders
    # get remaining quantities
    dfrq = remqty_nse(ib)

    # update target with remaining quantities
    dft = pd.read_pickle(fspath+'targets.pickle')

    # prepare the target
    dft = dft.assign(remqty=dft.symbol.map(dfrq.remqty.to_dict()))   # remaining quantities
    dft = dft[dft.remqty > 0].reset_index(drop=True) # remove blacklisted (remqty <= 0)

    # for first run, check if SOW orders are empty
    if not df_openord.empty:   # open orders are present due to BUY orders
        if df_openord[df_openord.action == 'SELL'].empty: # but there are no SELL orders
            print('SOW orders empty. First trades will be done.')
            dft = upd(ib, dft)
            dft.to_pickle(fspath+'targets.pickle')
            first_sell_blks = trade_blocks(ib, dft, exchange=exchange)

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # WARNING: THE FOLLOWING CODE PLACES FIRST 'SOW' SELL ORDERS
            # __________________________________________________________
            first_sell_trades = doTrades(ib, first_sell_blks)
            # @@@@@@@@    END OF ORDER PLACEMENT   @@@@@@@@@@@@@@@@@@

            util.df(first_sell_trades).to_pickle(fspath+'first_sell_trades.pickle')

    # .. get filled symbols
    fill_symbols = []
    if ib.fills():     # get fill symbols, if there are any SELL fills!
        df_fc = util.df(list(util.df(ib.fills())['contract'])) # filled contracts
        df_fc = df_fc[list(df_fc)[:6]]
        df_fc = df_fc.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})
        df_fc

        df_fe = util.df(list(util.df(ib.fills())['execution'])) # executed contracts
        df_fe = df_fe[list(df_fe)[4:13]].drop('liquidation', 1)

        df_fill = df_fc.join(df_fe)

        # get the fill symbols
        fill_symbols = list(df_fill.symbol.unique())

        # identify the recalc orders which needs to be cancelled and reordered
        df_rcalc = dft[dft.symbol.isin(fill_symbols)]

        if not df_rcalc.empty: # re-calculation has trades (genuine or from last day)
            if df_openord.empty: #...but openorder is empty. This is because of last day's trades still available.
                print('NOTE: Dynamic calculation is to be run only an hour before markets open.\n')
                print('Entire target SELL trade will be placed')
                sell_blks_wo_openord = trade_blocks(ib, dft, exchange=exchange)

                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                # WARNING: THE FOLLOWING CODE PLACES FIRST 'SOW' SELL ORDERS
                # __________________________________________________________
                sell_trades_wo_openord = doTrades(ib, sell_blks_wo_openord)
                # @@@@@@@@    END OF ORDER PLACEMENT   @@@@@@@@@@@@@@@@@@@@@
                return None # get out of the loop!

            # else it is a genuine recalc... which needs to be cancelled
            sell_ordId = df_openord[df_openord.action == 'SELL'].set_index('conId')[['orderId']].to_dict()['orderId']
            df_cancel_rc = df_rcalc.assign(ordId=df_rcalc.optId.astype('int32').map(sell_ordId)).dropna()
            cancelords = [o for o in ib.openOrders() if o.orderId in list(df_cancel_rc.ordId.astype(int))]
            rc_cancelled = [ib.cancelOrder(c) for c in cancelords]

            # recalc remaining quantities are 're-quantified'
            df_remq = remqty_nse(ib)
            df_rcalc = df_rcalc.assign(remqty=df_rcalc.symbol.map(df_remq.remqty.to_dict()))   # remaining quantities
            df_rcalc = df_rcalc[df_rcalc.remqty > 0].reset_index(drop=True) # remove blacklisted (remqty <= 0)

            # expected price is jacked up to cover for the risk
            df_rcalc = upd(ib, df_rcalc)
            df_rcalc = df_rcalc.assign(expPrice=get_prec(df_rcalc.expPrice*1.25,0.05))
            # expRom is also recalculated
            df_rcalc = df_rcalc.assign(expRom=df_rcalc.expPrice*df_rcalc.lot/df_rcalc.margin*365/df_rcalc.dte)

            # SOW trades are placed
            rcalc_blks = trade_blocks(ib, df_rcalc, exchange=exchange)

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # WARNING: THE FOLLOWING CODE PLACES 'SOW' SELL ORDERS
            # _______________________________________________________
            rcalc_trades = doTrades(ib, rcalc_blks)
            # @@@@@@@@    END OF ORDER PLACEMENT   @@@@@@@@@@@@@@@@@@


            # remove the recalc optIds from dft
            dft=dft[~dft.optId.isin(list(df_rcalc.optId))]

            # re-add df_rcalc to dft and pickle
            dft = grp_opts(pd.concat([dft, df_rcalc[dft.columns]]))

            # write to pickle
            dft.to_pickle((fspath+'targets.pickle'))
        
        return None

#_____________________________________

