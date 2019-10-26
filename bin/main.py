# header.py

# imports
from support import *

import time
from datetime import datetime
from itertools import product
from math import erf, sqrt
import sys

# start the ib loop

# connect to the market set in variables.json
try:
    if not ib.isConnected():
        ib = get_connected(market, 'live')
except Exception as e:
    ib = get_connected(market, 'live')

# from json
a = assign_var(market) + assign_var('common')
for v in a:
    exec(v)

# reset the log
with open(logpath+'main.log', 'w'):
    pass # clear the run log

util.logToFile(logpath+'main.log')
    
jup_disp_adjust() # adjust jupyter's display

#_____________________________________

# make_targets.py
def make_targets(margin_breached):
    
    '''Makes chains, ohlcs and targets
    Arg: (margin_breached) flag
    Returns: df_chains, df_ohlcsd and df_targets dataframes'''
    
    #... get the underlyings
    # timekeepers
    begin = time.time()  # for the overall program
    start = begin

    #..instruments
    und_contracts = get_instruments(ib, market)

#     # DATA LIMITER!!!
#     und_contracts = und_contracts[:25]

    id_sym = {u.conId: u.symbol for u in und_contracts}
    und_blks = [und_contracts[i: i+blk] for i in range(0, len(und_contracts), blk)]

    #..undprices
    und_ticks = ib.reqTickers(*und_contracts)
    undPrice = {u.contract.conId: u.marketPrice() for u in und_ticks}

    print(f"\nCompleted getting underlyings in {sec2hms(time.time()-start)}\n")

    #... get the chains
    start = time.time()

    # get the chains

    async def chains_coro(und_contracts):
        '''Get the chains for underlyings
        Arg: (und_contracts) as a list
        Returns: awaits of reqSecDefOptPramsAsyncs'''
        ch_tasks = [ib.reqSecDefOptParamsAsync(underlyingSymbol=c.symbol, futFopExchange='', 
                                 underlyingConId=c.conId, underlyingSecType=c.secType)
                                 for c in und_contracts]
        return await asyncio.gather(*ch_tasks)

    ch = ib.run(chains_coro(und_contracts))

    chs = [b for a in ch for b in a]

    chains = {c.underlyingConId: c for c in chs}

    sek = {b for a in [list(product([k], m.expirations, m.strikes)) for k, m in chains.items()] for b in a}

    df_chains1 = pd.DataFrame(list(sek), columns=['undId', 'expiry', 'strike'])
    df_chains1 = df_chains1.assign(undId=df_chains1.undId.astype('int32'))

    df_chains1 = df_chains1.assign(symbol = df_chains1.undId.map(id_sym), 
                                 undPrice = df_chains1.undId.map(undPrice))

    if market == 'nse':  # code for NSE only
        # add expiryM to get lots from get_nse_lots()
        df_chains1 = df_chains1.assign(expiryM=pd.to_datetime(df_chains1.expiry).dt.strftime('%Y-%m'))
        # update lots of NSE equities
        lots = get_nse_lots()
        lots = lots.assign(expiryM=lots.expiryM.astype('str'))
        lots = lots[['symbol', 'expiryM', 'lot']].set_index(['symbol', 'expiryM'])

        chz = df_chains1.set_index(['symbol', 'expiryM'])

        df_chains = chz.join(lots).reset_index().drop('expiryM', 1)

        # For those with nan - forget the expiry!
        df_symlot = lots.reset_index().drop('expiryM', 1).drop_duplicates()
        symlot = dict(zip(df_symlot.symbol, df_symlot.lot))

        df_chains = df_chains.assign(lot=df_chains.symbol.map(symlot).fillna(df_chains.lot))

    else: # For SNP
        df_chains = df_chains1.assign(lot=100)

    df_chains = df_chains.sort_values(['symbol', 'expiry', 'strike'])
    df_chains = df_chains[['symbol', 'undId', 'expiry', 'strike', 'undPrice', 'lot']].reset_index(drop=True)
    df_chains = df_chains.assign(dte=df_chains.expiry.apply(get_dte))

    df_chains.to_pickle(fspath+'chains.pkl')

    print(f"Completed getting chains in {sec2hms(time.time()-start)}\n")

    #... Get the OHLCs (asyncio)
    start = time.time()

    async def ohlc_coro(und_contracts):
        # build the tasks
        ohlc_tasks = [ib.reqHistoricalDataAsync(contract=qc, endDateTime='', 
                                            durationStr='365 D', barSizeSetting='1 day',  
                                                        whatToShow='Trades', useRTH=True) for qc in und_contracts]
        return await asyncio.gather(*ohlc_tasks)

    ohlcs = [ib.run(ohlc_coro(unds)) for unds in und_blks] 

    ohlcs = [i for j in ohlcs for i in j]

    # make the ohlc dataframe
    df_ohlc = pd.DataFrame()
    for i, o in enumerate(ohlcs):
        df = util.df(o)
        if not o:
            print(f'{und_contracts[i].symbol} ohlc is empty')
        else:
            df_ohlc = df_ohlc.append(df.assign(symbol=und_contracts[i].symbol))

    #... compute the standard deviations
    df_ohlc = df_ohlc.assign(date=pd.to_datetime(df_ohlc.date, format='%Y-%m-%d'))

    grp1 = df_ohlc.groupby('symbol')
    grp2 = grp1.apply(lambda df: df.sort_values('date', ascending=False)).reset_index(drop=True)

    df_ohlcsd = grp2.groupby('symbol').apply(lambda df: df.assign(stDev=df.close.expanding(1).std(ddof=0))).reset_index(drop=True)

    df_ohlcsd.to_pickle(fspath+'ohlcs.pkl')

    print(f"Completed getting ohlcs in {sec2hms(time.time()-start)}\n")

    if not margin_breached:
        # ... Size the chains
        start = time.time()

        try:
            df_chains.empty
        except NameError as e:
            df_chains = pd.read_pickle(fspath+'chains.pkl')

        try:
            df_ohlcsd.empty
        except NameError as e:
            df_ohlcsd = pd.read_pickle(fspath+'ohlcs.pkl')

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

        # remove the calls and puts near strike price
        c_mask = (df2.right == 'C') & (df2.strike > df2.undPrice + callstdmult*df2.stDev)
        p_mask = (df2.right == 'P') & (df2.strike < df2.undPrice - putstdmult*df2.stDev)
        df3 = df2[c_mask | p_mask].reset_index(drop=True)

        # make (df and dte) tuple for fallrise
        tup4fr = [(df_ohlcsd[df_ohlcsd.symbol == s.symbol], s.dte) 
                  for s in df3[['symbol', 'dte']].drop_duplicates().itertuples()]

        # get the fallrise and put it into a dataframe
        fr = [fallrise(*t) for t in tup4fr]
        df_fr = pd.DataFrame(fr, columns=['symbol', 'dte', 'fall', 'rise' ])

        # merge with options df
        df3 = pd.merge(df3, df_fr, on=['symbol', 'dte'])

        # make reference strikes from fall_rise
        df3 = df3.assign(strikeRef = np.where(df3.right == 'P', 
                                                    df3.undPrice-df3.fall, 
                                                    df3.undPrice+df3.rise))
        # get lo52 and hi52
        df3 = df3.set_index('symbol').join(df_ohlcsd.groupby('symbol')
                                                 .close.agg(['min', 'max'])
                                                 .rename(columns={'min': 'lo52', 'max': 'hi52'})).reset_index()

        # ...Filter # 1: Top nband for Puts and Calls > mindte
        # building SELLS for further expiries
        df4 = df3[df3.dte >= mindte]

        gb = df4.groupby(['right'])

        if 'C' in [k for k in gb.indices]:
            df_calls = gb.get_group('C').reset_index(drop=True).sort_values(['symbol', 'dte', 'strike'], ascending=[True, True, True])
            df_calls = df_calls.groupby(['symbol', 'dte']).head(nBand)
        else:
            df_calls = pd.DataFrame([])

        if 'P' in [k for k in gb.indices]:
            df_puts = gb.get_group('P').reset_index(drop=True).sort_values(['symbol', 'dte', 'strike'], ascending=[True, True, False])
            df_puts = df_puts.groupby(['symbol', 'dte']).head(nBand)
        else:
            df_puts =  pd.DataFrame([])

        df5 = pd.concat([df_puts, df_calls]).reset_index(drop=True)

        # ....Filter # 2: nBands with fallrise for expiries less than a week
        # building SELLS for nearer expiries with fallrise

        df6 = df3[df3.dte < mindte]

        if df6.empty: # there are no contracts below minimum dte
            df8 = pd.DataFrame([])
        else:
            # get the strikes closest to the reference strikes
            df7 = df6.groupby(['symbol', 'dte'], as_index=False) \
                     .apply(lambda g: g.iloc[abs(g.strike - g.strikeRef) \
                     .argsort()[:nBand]]) \
                     .reset_index(drop=True)

            df8 = df7.sort_values(['symbol', 'dte', 'strike'])

        # make the target master
        df9 = pd.concat([df8, df5], sort=False).sort_values(['symbol', 'dte', 'strike'], 
                                                            ascending=[True, True, False]).drop_duplicates().reset_index(drop=True)

        # Based on filter selection in json....
        if callRise:
            df9 = df9[~((df9.right == 'C') & (df9.strike < df9.strikeRef))].reset_index(drop=True)

        if putFall:
            df9 = df9[~((df9.right =='P') & (df9.strike > df9.strikeRef))].reset_index(drop=True)

        if onlyPuts:
            df9 = df9[df9.right == 'P'].reset_index(drop=True)

        print(f"Completed sizing for targetting in {sec2hms(time.time()-start)}\n")

        #... Qualify the option contracts
        start = time.time()

        qo_task = []
        opts = [Option(i.symbol, i.expiry, i.strike, i.right, exchange) for i in df9[['symbol', 'expiry', 'strike', 'right']].itertuples()]

        qual_opts = ib.qualifyContracts(*opts)

        print(f"Completed qualifying option contracts in {sec2hms(time.time()-start)}\n")

        #... Get tickers and optPrice (asyncio)
        start = time.time()
        opt_blks = [qual_opts[i: i+blk] for i in range(0, len(qual_opts), blk)]

        async def optpricecoro(opt):
            tasks = [ib.reqTickersAsync(c) for c in opt]
            return await asyncio.gather(*tasks)

        ot = [i for j in [ib.run(optpricecoro(opts)) for opts in opt_blks] for i in j]

        opt_price = {t.contract.conId: t.marketPrice() 
                     for opts in ot 
                     for t in opts 
                     if t.marketPrice() > -1} # cleaned nans

        df_opt1 = util.df(qual_opts)
        df_opt2 = df_opt1[list(df_opt1)[1:6]].rename(columns={'lastTradeDateOrContractMonth': 'expiry', 'conId': 'optId'})

        df_opt3 = df_opt2.assign(optPrice=df_opt2.optId.map(opt_price))
        df_opt4 = df_opt3[df_opt3.optPrice >= 0]

        df_opt5 = pd.merge(df_opt4, df9, on=['symbol', 'expiry', 'strike', 'right'])
        df_opt5 = df_opt5.dropna() # remove NAs in - for instance from lots

        print(f"Completed getting option prices in {sec2hms(time.time()-start)}\n")

        #... Get the margins
        start = time.time()

        # build options and orders
        mgn_opts = list(df_opt5.optId.map({c.conId: c for c in qual_opts}))

        if market == 'snp':
            mgn_ords = [Order(action='SELL', orderType='MKT', totalQuantity=1, whatIf=True) for r in df_opt5.lot]
        else:
            mgn_ords = [Order(action='SELL', orderType='MKT', totalQuantity=r, whatIf=True) for r in df_opt5.lot]

        co = list(zip(mgn_opts, mgn_ords))

        async def margin_coro(co):   
            tasks = [ib.whatIfOrderAsync(*c) for c in co]
            return await asyncio.gather(*tasks)

        margins = ib.run(margin_coro(co))

        # market is checked to weed out wierd commissions. NSE has commission, while SNP has maxCommission!
        if market == 'nse':
            df_opt6 = df_opt5.assign(margin=[catch(lambda: float(m.initMarginChange)) for m in margins], 
                               comm=[catch(lambda: float(m.commission)) for m in margins])
        else:
            df_opt6 = df_opt5.assign(margin=[catch(lambda: float(m.initMarginChange)) for m in margins], 
                               comm=[catch(lambda: float(m.maxCommission)) for m in margins])    

        df_opt7 = df_opt6[df_opt6.margin < 1.7e7] # remove too high margins

        df_opt8 = df_opt7.assign(PoP=[erf(i/sqrt(2.0)) for i in abs(df_opt7.strike-df_opt7.undPrice)/df_opt7.stDev],
                                RoM = abs((df_opt7.optPrice*df_opt7.lot-df_opt7.comm)/df_opt7.margin*365/df_opt7.dte))

        df_opt8.to_pickle(fspath+'opts.pkl')

        print(f"Completed getting margins, RoM and PoP in {sec2hms(time.time()-start)}\n")

        # ... Eliminate < 0 remaining quantities
        start = time.time()

        try:
            df_opt8.empty
            df_opt9 = df_opt8
        except NameError as e:
            df_opt9 = pd.read_pickle(fspath+'opts.pkl')
            df_chains = pd.read_pickle(fspath+'chains.pkl')

        dfrq1 = dfrq(ib, df_chains, exchange) # get the remaining quantities

        # integrate df_opt with remaining quantity

        df_opt10 = df_opt9.set_index('symbol').join(dfrq1).reset_index()
        df_opt10 = df_opt10[df_opt10.remq > 0]  # remove options that have busted the remaining quantities

        df_opt11 = df_opt10[~df_opt10.symbol.isin(blacklist)] # remove blacklists

        # set the expected price and expected rom
        df_opt12 = df_opt11.assign(qty=1, 
                                   expPrice=np.maximum( \
                                                     np.maximum(minexpRom/df_opt11.RoM*(df_opt11.optPrice+prec), minexpOptPrice), \
                                                              df_opt11.optPrice+prec))
        df_opt13 = df_opt12.assign(expRom=(df_opt12.expPrice*df_opt12.lot-df_opt12.comm)/df_opt12.margin*365/df_opt12.dte)                       

        # Re-adjust expPrice for expRom < minexpRom. 
        # Re-calculate expRom.
        # This adjustment is for optPrice = 0 and negative margin options.
        mask = df_opt13.expRom < minexpRom

        df_opt13[mask] = df_opt13[mask].assign(expPrice=minexpRom/df_opt13[mask].expRom*df_opt13[mask].expPrice)

        df_opt13 = df_opt13.replace([np.inf, -np.inf], np.nan).dropna() # remove infinities

        df_opt13.loc[df_opt13.expPrice <= 0, 'expPrice'] = minexpOptPrice # set minimum expected option price

        mask = df_opt13.expRom < minexpRom # to correct the length of the df

        df_opt13 = df_opt13.assign(expPrice=[get_prec(p, prec) for p in df_opt13.expPrice])
        df_opt13[mask] = df_opt13[mask].assign(expRom=(df_opt13[mask].expPrice*df_opt13[mask].lot-df_opt13[mask].comm)/df_opt13[mask].margin*365/df_opt13[mask].dte)

        cols = ['symbol', 'optId', 'expiry', 'strike', 'right', 'dte', 'undId', 'undPrice', 'lot', 'stDev', 'fall', 'rise', 
                'strikeRef', 'lo52', 'hi52', 'margin', 'comm', 'PoP', 'RoM', 'remq', 'qty', 'optPrice', 'expPrice', 'expRom']
        df_opt13 = df_opt13[cols]
        
        # symbols busting remaining quantity limit
        d = {'qty': 'sumOrdQty', 'remq': 'remq'}
        df_bustingrq = df_opt13.groupby('symbol').agg({'qty': 'sum', 'remq': 'mean'}).rename(columns=d)
        df_bustingrq = df_bustingrq[df_bustingrq.sumOrdQty > df_bustingrq.remq].reset_index()

        df_bustingrq.assign(delta=df_bustingrq.remq.sub(df_bustingrq.sumOrdQty, axis=0)).sort_values('delta')

        df_opt13.to_pickle(fspath+'targets.pkl')

        df_targets = pd.read_pickle(fspath+'targets.pkl').reset_index(drop=True)

        print(f"...Created targets. COMPLETE program took {sec2hms(time.time()-begin)}...\n")
    
    else:
        print(f"!!! Margin Breach for {market}. No targets generated !!!\n")
        df_targets = pd.read_pickle('./templates/df_trades.pkl')
    
    return df_targets

# ui_select.py
#... user interface
def ask_user():
    '''Asks user what needs to be done
    Arg: None
    Returns: (int) between 0..7'''
    
    # Get user input
    askmsg = "\nChoose from the following numbers:\n" + \
             "1) Delete ALL logs and data, generate fresh targets and covers\n" + \
             "2) Place ALL (Buy and Sell) trades\n" + \
             "3) Only place closing BUY trades\n\n" + \
             "0) ABORT\n\n"
    
    while True:
        try:
            ip = int(input(askmsg+'\n'))
        except ValueError:
            print("\nSorry, I didn't understand what you entered. Try again!\n")
            continue # Loop again
        if not ip in [0, 1, 2, 3]:
            print(f"{ip} is a wrong number! Choose any number from 0 to 4 and press <Enter>\n")
        else:
            break # success and exit loop
    
    return ip    

# branching.py
#...from the selected user inputs...
if __name__ == '__main__':
    userip = ask_user()
    
    # check for available funds
    acsum = get_acc_summary(ib)

    if acsum['avFunds'] < avlblMarginlmt:
        print(f"Fresh SELLs will not be placed due to low available funds\n{acsum['avFunds']} < {avlblMarginlmt}!!\n")
        margin_breached = True
    else:
        margin_breached = False

    if userip == 0: # Exit
        print("\n....ABORTING....\n")
        sys.exit(1)
        
    if userip == 1: # Delete all logs+data, generate targets and covers
        print(f"....Generating FRESH targets for {market.upper()}...\n")
        
        delete_all_data(market)
        df_targets = make_targets(margin_breached).reset_index(drop=True)
        
        df_chains = pd.read_pickle(fspath+'chains.pkl')
        df_ohlcsd = pd.read_pickle(fspath+'ohlcs.pkl')
        df_covered = covers(ib, market, df_chains, df_ohlcsd, fspath)

        print(f"....Completed making fresh targets for {market.upper()}...\n")
    
    if userip == 2: # Place buys and sells
              
        begin = time.time()
        
        print("\n...Placing ALL sells and closing buys...\n")
        
        # cancel existing all open trades
        ib.reqGlobalCancel()

        # get targets from the pickle if fresh, else generate
        tgt_hrs = (datetime.now() - datetime.fromtimestamp(path.getmtime(fspath+'targets.pkl'))).total_seconds()/60/60
        if tgt_hrs > 3: # needs target regeneration
            df_targets = make_targets().reset_index(drop=True) # regenerate targets
        else:
            df_targets=pd.read_pickle(fspath+'targets.pkl').reset_index(drop=True) 
            
        # place the sell trades
        if not df_targets.empty:
            sell_tb = sells(ib, df_targets, exchange)
            sell_trades = doTrades(ib, sell_tb)
        else:
            print("\n...No targets to SELL...\n")
        
        #... Place buys from existing open trades
        df_buys = get_df_buys(ib, market, prec)
        if not df_buys.empty:
            buy_tb = buys(ib, df_buys, exchange)
            buy_trades = doTrades(ib, buy_tb)
        else:
            print(f"No options to close in {market.upper()}\n")
        
        #... Place covered SELL trades
        df_chains = pd.read_pickle(fspath+'chains.pkl')
        df_ohlcsd = pd.read_pickle(fspath+'ohlcs.pkl')
        df_covered = covers(ib, market, df_chains, df_ohlcsd, fspath)
        if not df_covered.empty:
            sell_covers = sells(ib, df_covered, exchange)
            covertrades = doTrades(ib, sell_covers)
        else:
            print(f"No covering contracts needed / available on {market.upper()} longs and shorts!\n")
              
        print(f"Completed placing ALL sells and buys in {sec2hms(time.time()-begin)} for {market.upper()}\n")
        
    if userip == 3: # Only closing buys - dynamic trade
        print("\n...Placing closing buys only...\n")
        
        #... Place buys from existing open trades
        df_buys = get_df_buys(ib, market, prec)
        if not df_buys.empty:
            buy_tb = buys(ib, df_buys, exchange)
            buy_trades = doTrades(ib, buy_tb)
        else:
            print(f"No buys on existing {market.upper()} open trades\n")

#_____________________________________

