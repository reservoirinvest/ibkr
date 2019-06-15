# nse_main.py
# Main program for nse

from helper import *
from nse_func import *



# Do assignments
a = assign_var('nse')
for v in a:
    exec(v)
    
ip = 1

def do_an_opt(ib, row, df_l):
    '''Gets the options for a contract
    Args:
        (ib) as connection object
        (row) as data series row from df_l
        (df_l) for expiries with different lots check
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

    # weed dtes greater than maxdte
    df_opt = df_opt[df_opt.dte.between(1, maxdte)].reset_index(drop=True)

    # weed option contracts within standard deviations band
    df_hist = catch(lambda: do_hist(ib, row.undId))

    if type(df_hist) != float: # df_hist is not nan from catch!

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

        # get the lots from dte
        dtelot_dict = df_l[df_l.symbol == row.symbol].set_index('dte').lot.to_dict() # {dte: lot}

        df_opt = df_opt.assign(lot=df_opt.dte.map(dtelot_dict).fillna(df_opt.lot)).reset_index(drop=True)

        # get the underlying margin from closest strike
        df_opt = closest_margin(ib, df_opt, exchange)

        #... get the margins

        # build contracts and orders
        opt_con_dict = {c.conId: c for c in contracts if c.conId in list(df_opt.optId)}
        opt_ord_dict = {i: Order(action='SELL', orderType='MKT', totalQuantity=lot) for i, lot in zip(df_opt.optId, df_opt.lot)}

        # combine contracts and orders
        opts = zip(df_opt.optId.map(opt_con_dict), df_opt.optId.map(opt_ord_dict))

        # gather and get the margins through asyncio
        m = []
        async def coro():
            for c, o in opts:
                m.append(await getMarginAsync(ib, c, o))
            return m
            
        opt_mgn_dict = {k: v for d in asyncio.run(coro()) for k, v in d.items()}

        df_opt = df_opt.assign(optMargin=df_opt.optId.map(opt_mgn_dict)).fillna(df_opt.undMargin)

         # ... get the prices
        opt_tickers = ib.reqTickers(*contracts)
        optPrices = {t.contract.conId: t.marketPrice() for t in opt_tickers}

        df_opt = df_opt.assign(optPrice=df_opt.optId.map(optPrices))
        df_opt = df_opt.assign(rom=df_opt.optPrice/df_opt.optMargin*365/df_opt.dte*df_opt.lot).sort_values('rom', ascending=False).reset_index(drop=True)

        # get the 52 week highs and lows
        df_opt = df_opt.assign(lo52 = df_hist.iloc[:tradingdays].low.min(), hi52 = df_hist.iloc[:tradingdays].high.max())

        #... get the maximum fall rise for the dtes

        dte_list = sorted(list(df_opt.dte.unique()))
        fr_dict = [fallrise(df_hist, dte) for dte in dte_list]

        df_opt = df_opt.join(pd.DataFrame(list(df_opt.dte.map({k: v for i in fr_dict for k, v in i.items()}))))

        df_opt.to_pickle(fspath+'_'+row.symbol+'.opt')
    
        return df_opt
    
    else: # df_hist failed, so cannot generate df_opt
        return np.nan

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
    return {c.conId:float(aw.initMarginChange)}


# do the appropriate function
if ip in [1, 2]:    #  build ALL the options
    
    with get_connected('nse', 'live') as ib:
        
        with open(logpath+'test.log', 'w'):
            pass # clear the run log
        
        util.logToFile(logpath+'test.log')
        
        logging.basicConfig(level=logging.DEBUG)
        
        util.logging.info('####                     NSE BUILD STARTED                  ####')
        print(f'NSE options build started at {datetime.datetime.now()}...')
        s = time.perf_counter()
        
        # generate the symexplots
        # df_l = symexplots(ib)
        df_l = pd.read_pickle(fspath+"symexplots.pickle").sort_values('symbol')
        
        # get the series of rows for options
        rows = [s for i, s in df_l.iterrows()]
        
        if ip is 2:
            # remove rows already pickled
            pkl_done = pd.read_pickle(fspath+'opts.pickle').symbol.unique()
            rows = [row for row in rows if row.symbol not in pkl_done]
        

        # get the options
        tqr = trange(len(rows), desc='Processing', leave=True) # Initializing tqdm
        opts = [] # initializing list of opts

        for row in rows:
            tqr.set_description(f"Processing [{row.symbol}]")
            tqr.refresh() # to show immediately the update
            opts.append([do_an_opt(ib, row, df_l)])
            tqr.update(1)
        tqr.close()