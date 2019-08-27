# sized_nse.py

"""Program to size nse options
Date: 23-June-2019
Ver: 1.0
Time taken: 40 mins
"""

from z_helper import *

from chains_nse import *
from ohlcs import *

# # from json
a = assign_var('common') + assign_var('nse')
for v in a:
    exec(v)

def sized_nse(ib, df_chains, df_ohlcs):
    '''Generates sized nse pickle
    Args:
        (ib) as connection object
        (df_chains) from pickled / generated df_chains
        (df_ohlcs) from pickled / generated df_ohlcs
    Returns:
        (df_optg) as group of sized df_opts. Also pickles'''
    
    # log to size_chains.log
    with open(logpath+'size_chains.log', 'w'):
        pass # clear the run log
    util.logToFile(logpath+'size_chains.log')

    #... Remove chains not meeting put and call std filter

    # get dte and remove those greater than maxdte
    df_chains = df_chains.assign(dte=df_chains.expiry.apply(get_dte))                    
    df_chains = df_chains[df_chains.dte <= maxdte]
    
    # replace dte with 1 for dte <= 0
    df_chains.loc[df_chains.dte <=0,  'dte'] = 1

    # generate std dataframe
    df = df_ohlcs[['symbol', 'stDev']]  # lookup dataframe
    df = df.assign(dte=df.groupby('symbol').cumcount()) # get the cumulative count for location as dte
    df.set_index(['symbol', 'dte'])

    df1 = df_chains[['symbol', 'dte']]  # data to be looked at
    df2 = df1.drop_duplicates()  # remove duplicates

    df_std = df2.set_index(['symbol', 'dte']).join(df.set_index(['symbol', 'dte']))

    # join to get std in chains
    df_chainstd = df_chains.set_index(['symbol', 'dte']).join(df_std).reset_index()

    und_contracts = [Stock(symbol, exchange=exchange) for symbol in df_chainstd.symbol.unique()]

    und_quals = ib.qualifyContracts(*und_contracts)
    tickers = ib.reqTickers(*und_quals)

    uprice_dict = {u.contract.conId: u.marketPrice() for u in tickers}

    df_chainstd = df_chainstd.assign(undPrice=df_chainstd.undId.astype('int').map(uprice_dict))

    # make puts and calls dataframe with std filter
    df_puts = df_chainstd[df_chainstd.strike < (df_chainstd.undPrice-(df_chainstd.stDev*putstdmult))]
    df_puts = df_puts.assign(right = 'P')

    df_calls = df_chainstd[df_chainstd.strike > (df_chainstd.undPrice+(df_chainstd.stDev*callstdmult))]
    df_calls = df_calls.assign(right = 'C')

    df_opt = pd.concat([df_puts, df_calls], sort=False).reset_index(drop=True)

    # get lo52 and hi52
    df_opt = df_opt.set_index('symbol').join(df_ohlcs.groupby('symbol')
                                             .close.agg(['min', 'max'])
                                             .rename(columns={'min': 'lo52', 'max': 'hi52'})).reset_index()

    # make (df and dte) tuple for fallrise
    tup4fr = [(df_ohlcs[df_ohlcs.symbol == s.symbol], s.dte) 
              for s in df_opt[['symbol', 'dte']].drop_duplicates().itertuples()]

    # get the fallrise and put it into a dataframe
    fr = [fallrise(*t) for t in tup4fr]
    df_fr = pd.DataFrame(fr, columns=['symbol', 'dte', 'fall', 'rise' ])

    # merge with df_opt
    df_opt = pd.merge(df_opt, df_fr, on=['symbol', 'dte'])

    #####!!! TEMPORARY CODE BLOCK !!!######
    
#     df_opt = df_opt.assign(strikeRef = np.where(df_opt.right == 'P', 
#                                                 df_opt.undPrice-df_opt.stDev*putstdmult, 
#                                                 df_opt.undPrice+df_opt.rise*callstdmult))      

    ##### END TEMPORARTY CODE BLOCK ######

    # make reference strikes from fall_rise
    df_opt = df_opt.assign(strikeRef = np.where(df_opt.right == 'P', 
                                                df_opt.undPrice-df_opt.fall, 
                                                df_opt.undPrice+df_opt.rise))

    # get the strikes closest to the reference strikes
    df_opt = df_opt.groupby(['symbol', 'dte']) \
                             .apply(lambda g: g.iloc[abs(g.strike-g.strikeRef) \
                                                     .argsort()[:nBand]])

    df_opt = df_opt.set_index('symbol').reset_index()

    # get the option contracts
    opt_list = [Option(i.symbol, i.expiry, i.strike, i.right, exchange) for i in df_opt[['symbol', 'expiry', 'strike', 'right']].itertuples()]

    opt_contracts = []

    print(f"\nQualifying {len(opt_list)} option contracts ...\n")
    opt_contracts = ib.qualifyContracts(*opt_list)

    # integrate optId with df_opt and remove df_opt without optId
    dfq = util.df(opt_contracts).iloc[:, 1:6]
    dfq.columns=['optId', 'symbol', 'expiry', 'strike', 'right'] # rename columns
    df_opt=df_opt.merge(dfq, on=['symbol', 'expiry', 'strike', 'right'], how='left')
    df_opt = df_opt[~df_opt.optId.isnull()]
    df_opt = df_opt.assign(optId=df_opt.optId.astype('int'))

    # get the option prices

    ticker = ib.reqTickers(*opt_contracts)

    df_prices = pd.DataFrame({t.contract.conId: {'bid':t.bid, 'ask':t.ask, 'close':t.close} for t in ticker}).T

    # ...get margins

    # prepare the lots
    idlot_idx = df_opt[['optId', 'lot']].set_index('optId').to_dict('index')
    idlot = {k: Order(action='SELL', orderType='MKT', totalQuantity=v['lot'], whatIf=True) for k, v in idlot_idx.items()}

    co = [(c, idlot[c.conId]) for c in opt_contracts]

    # co = co[:110]  # DATA LIMITER !!!
    coblks = [co[i: i+blk] for i in range(0, len(co), blk)]

    m = {} # empty dictionary to collect outputs of getMarginAsync

    async def coro(coblk):
        with tqdm(total=len(coblk), file=sys.stdout, unit=' symexpiry') as tqm:
            for c, o in coblk:
                tqm.set_description(f"IBKR margins for  {c.localSymbol.ljust(22)}")
                m.update(await getMarginAsync(ib, c, o))
                tqm.update(1)
            return m

    # run co-routines to get the margins        
    for coblk in coblks:
        asyncio.run(coro(coblk))

    # put margins to df_opt
    m_dict = {i: float(j.initMarginChange) for i, j in {k: v for k, v in m.items() if v}.items() if str(j) != 'nan'}

    df_margin = pd.DataFrame.from_dict(m_dict, orient='index', columns=['margin'])

    df_opt = df_opt.set_index('optId').join(df_prices).join(df_margin).reset_index()

    df_opt = grp_opts(df_opt.assign(rom=df_opt.close/df_opt.margin*365/df_opt.dte*df_opt.lot))

    df_opt.to_pickle(fspath+'sized_nse.pkl')
    grp_opts(df_opt).to_excel(fspath+'sized_nse.xlsx', index=False, freeze_panes=(1,2))
       
    return df_opt

#_____________________________________

