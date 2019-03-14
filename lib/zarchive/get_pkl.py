def get_pkl(ib, contract):
    '''Function to pickle ohlc, underlying and options
    Arg: (contract) as the qualified contract object
    Returns: None, but pickles needed dataframes'''
    
    #... get ohlc, with cumulative volatality and standard deviation
    #_______________________________________________________________

    df_ohlc = get_hist(ib, contract, 365).set_index('date').sort_index(ascending = False)

    # get cumulative standard deviation
    df_stdev = pd.DataFrame(df_ohlc['close'].expanding(1).std(ddof=0))
    df_stdev.columns = ['stdev']

    # get cumulative volatility
    df_vol = pd.DataFrame(df_ohlc['close'].pct_change().expanding(1).std(ddof=0)*sqrt(tradingdays))
    df_vol.columns = ['volatility']

    df_ohlc1 = df_ohlc.join(df_vol)

    df_ohlc2 = df_ohlc1.join(df_stdev)

    #pickle the ohlc
    df_ohlc2.to_pickle(fspath+contract.symbol+'_ohlc.pkl')

    #... get the underlyings
    #_______________________

    ticker = get_dividend_ticker(ib, contract)

    df_und = util.df([ticker])

    cols = ['contract', 'time', 'bid', 'bidSize', 'ask', 'askSize', 'last', 'lastSize', 
            'volume', 'open', 'high', 'low', 'close', 'dividends']
    df_und = df_und[cols]

    df_und = df_und.assign(undPrice=np.where(df_und['last'].isnull(), df_und.close, df_und['last']))

    try: 
        divrate = df_und.dividends[0][0]/df_und.dividends[0][0]/df_und.undPrice
    except (TypeError, AttributeError) as e:
        divrate = 0.0

    df_und = df_und.assign(divrate=divrate)

    df_und = df_und.assign(symbol=[c[1].symbol for c in df_und.contract.items()])

    #... get the lot, margin, undPrice and dividend rate
    undlot = 100

    # margin of underlying
    order = Order(action='SELL', totalQuantity=undlot, orderType='MKT')

    margin = float(ib.whatIfOrder(contract, order).initMarginChange)

    df_und['margin'] = margin

    df_und.to_pickle(fspath+contract.symbol+'_und.pkl')

    #... get the options
    #___________________

    # symbol
    symbol = contract.symbol

    # rights
    right = ['P', 'C'] 

    # chains
    chains = ib.reqSecDefOptParams(underlyingSymbol=contract.symbol, futFopExchange='', 
                          underlyingConId=contract.conId, underlyingSecType=contract.secType)

    chain = next(c for c in chains if c.exchange == exchange)

    undPrice = df_und.undPrice[0]

    # get the strikes
    strikes = sorted([strike for strike in chain.strikes])

    # limit the expirations to between min and max dates
    expirations = sorted([exp for exp in chain.expirations 
                          if mindte < get_dte(exp) < maxdte])

    rights = ['P', 'C']

    df_tgt = pd.DataFrame([i for i in itertools.product([symbol], expirations, strikes, rights)], columns=['symbol', 'expiry', 'strike', 'right'])
    df_tgt['dte'] = [get_dte(e) for e in df_tgt.expiry]

    df_tgt1 = filter_kxdte(df_tgt, df_ohlc2, ohlc_min_dte)

    # make the contracts
    contracts = [Option(symbol, expiry, strike, right, exchange) 
                 for symbol, expiry, strike, right 
                 in zip(df_tgt1.symbol, df_tgt1.expiry, df_tgt1.strike, df_tgt1.right)]

    qc = [ib.qualifyContracts(*contracts[i: i+blks]) for i in range(0, len(contracts), blks)]

    qc1 = [q for q1 in qc for q in q1]
    df_qc = util.df(qc1).iloc[:, [2,3,4,5]]
    df_qc.columns=['symbol', 'expiry', 'strike', 'right']

    df_opt = df_qc.merge(df_tgt, on=list(df_qc), how='inner')
    df_opt['option'] = qc1

    df_und1 = df_und[['symbol', 'undPrice', 'margin']].set_index('symbol') # get respective columns from df_und
    df_und1['lot'] = 100

    df_opt = df_opt.set_index('symbol').join(df_und1) # join for lot and margin

    # get the standard deviation based on days to expiry
    df_opt = df_opt.assign(stdev=[df_ohlc2.iloc[i].stdev for i in df_opt.dte])

    # get the volatality based on days to expiry
    df_opt = df_opt.assign(volatility=[df_ohlc2.iloc[i].volatility for i in df_opt.dte])

    # high52 and low52 for the underlying
    df_opt = df_opt.assign(hi52 = df_ohlc2[:252].high.max())
    df_opt = df_opt.assign(lo52 = df_ohlc2[:252].low.min())
    df_opt.loc[df_opt.right == 'P', 'hi52'] = np.nan
    df_opt.loc[df_opt.right == 'C', 'lo52'] = np.nan

    df_opt.loc[df_opt.dte <= 1, 'dte'] = 2 # Make the dte as 2 for 1 day-to-expiry to prevent bsm divide-by-zero error

    # get the black scholes delta, call and put prices
    bsms = [get_bsm(undPrice, strike, dte, rate, volatility, divrate) 
            for undPrice, strike, dte, rate, volatility, divrate in 
            zip(itertools.repeat(undPrice), df_opt.strike, df_opt.dte, itertools.repeat(rate), df_opt.volatility, itertools.repeat(divrate))]

    df_bsm = pd.DataFrame(bsms)

    df_opt = df_opt.reset_index().join(df_bsm) # join with black-scholes

    df_opt['bsmPrice'] = np.where(df_opt.right == 'P', df_opt.bsmPut, df_opt.bsmCall)
    df_opt['pop'] = np.where(df_opt.right == 'C', 1-df_opt.bsmDelta, df_opt.bsmDelta)
    df_opt = df_opt.drop(['bsmCall', 'bsmPut', 'bsmDelta'], axis=1)

    # get the option prices
    cs = list(df_opt.option)

    # [catch(lambda: ib.reqTickers(c).marketPrice()) for i in range(0, len(cs), 100) for c in cs[i: i+100]]
    tickers = [ib.reqTickers(*cs[i: i+100]) for i in range(0, len(cs), 100)]

    df_opt = df_opt.assign(price=[t.marketPrice() for ts in tickers for t in ts])

    df_opt = df_opt.assign(rom=df_opt.price/df_opt.margin*tradingdays/df_opt.dte*df_opt.lot)

    df_opt.to_pickle(fspath+contract.symbol+'_opt.pkl')    
    
    return None