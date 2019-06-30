# chains_nse.py
"""Program generates symbols and lots for NSE
Date: 23-June-2019
Ver: 1.0
Time taken: 1 min 10 secs
"""

from z_helper import *

# from json
a = assign_var('nse') + assign_var('common')
for v in a:
    exec(v)

#...nse specific functions
#..........................

def get_lots():
    '''Get lots with expiry dates from nse csv
    Arg: None
    Returns: lots dataframe with expiry as YYYYMM'''

    url = 'https://www.nseindia.com/content/fo/fo_mktlots.csv'
    req = requests.get(url)
    data = StringIO(req.text)
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
    lots_df = lots_df.assign(expiryM=pd.to_datetime(lots_df.expiryM, format='%b-%y').dt.to_period('M'))

    # convert lots to integers
    lots_df = lots_df.assign(lot=pd.to_numeric(lots_df.lot, errors='coerce'))
    
    # convert & to %26
    lots_df = lots_df.assign(symbol=lots_df.symbol.str.replace('&', '%26'))

    return lots_df.reset_index(drop=True)

def get_xu(symbol: str) -> pd.DataFrame():
    '''Scrapes the symbols, expiry, undPrice from nse website'''
    # get expiries for the symbol
    url = 'https://www.nseindia.com/live_market/dynaContent/live_watch/option_chain/optionKeys.jsp?symbol='
    xpd = "//*[@id='date']" # xpath for date select options
    xpu = "//*[@id='wrapper_btm']/table[1]/tr/td[2]/div/span[1]/b" # xpath for undPrice
    
    res = requests.get(url + symbol).text
    htree = html.fromstring(res) #html is from lxml 
    expiries = [opt.text for e in htree.xpath(xpd) for opt in e if 'Select' not in opt.text.strip('')]
    undPrice = [float(e.text.split(' ')[1]) for e in htree.xpath(xpu)][0]

    # convert above to a DataFrame
    df = pd.DataFrame(list(product([symbol], expiries, [str(undPrice)])), 
                      columns=['symbol', 'expiry', 'undPrice'])

    return df.apply(pd.to_numeric, errors = 'ignore')

def get_nse_chain(symbol: str, expiry: 'datetime64', undPrice: float, lot: int) -> pd.DataFrame:
    '''scrapes one option chain from nse website pages'''
    
    url = 'https://www.nseindia.com/live_market/dynaContent/live_watch/option_chain/optionKeys.jsp?symbol='
    
    u = url+symbol+'&date='+expiry

    chainhtml = requests.get(u, headers=headers).content
    chain = pd.read_html(chainhtml)[1][:-1]  # read the first table and drop the total
    chain.columns=chain.columns.droplevel(0) # drop the first row of the header
    chain = chain.drop('Chart', 1)           # drop the charts
    
    cols = ['cOI', 'cOI_Chng', 'cVolume', 'cIV', 'cLTP', 'cNetChng', 'cBidQty', 'cBid', 'cAsk', 'cAskQty',
             'strike', 'pBidQty', 'pBid', 'pAsk', 'pAskQty', 'pNetChng', 'pLTP', 'pIV', 'pVolume', 'pOI_Chng', 'pOI']

    # rename the columns
    chain.columns = cols

    chain = chain.iloc[2:] # remove the first two rows

    # convert all to numeric
    chain = chain.apply(pd.to_numeric, errors = 'coerce')
    chain.insert(0, 'symbol', symbol)
    chain.insert(1, 'expiry', datetime.datetime.strptime(expiry, '%d%b%Y').date())
    chain.insert(2, 'undPrice', undPrice)
    chain.insert(3, 'lot', lot)
    
    return chain

def nse_chains():
    '''Gets nse chains with symbols and lots from nse web
    Arg: None
    Returns: df_chains as dataframe of option chains'''
    
    #.... Symbols & Lots
    #.....................
    
    df_lots = get_lots()
    symbols = sorted(list(df_lots.symbol.unique()))
#     symbols = [s for s in symbols if s in ['NIFTY', 'PNB']] # DATA LIMITER!!!

    # get the strikes, expiry and undPrices
    sxu = []
    with tqdm(total=len(symbols), file=sys.stdout, unit=" symbol") as tqs:
        for s in symbols:
            tqs.set_description(f"Price and expiry from NSE web  {s[:9].ljust(9)}", refresh=True)
            sxu.append(get_xu(s))
            tqs.update(1)
            
    # get the strikes, expiry and undPrices
    df_sxu = pd.concat(sxu).reset_index(drop=True)

    df_sxu = df_sxu.assign(expiry=pd.to_datetime(df_sxu.expiry))

    # get the lots
    df_sxul = df_sxu.assign(expiryM=df_sxu.expiry.dt.to_period('M')).merge(df_lots).drop('expiryM', 1)

    # convert expiry to nse friendly date
    df_sxul = df_sxul.assign(expiry=[f"{dt.day}{calendar.month_abbr[dt.month].upper()}{dt.year}" for dt in df_sxul.expiry])

    chains = []
    with tqdm(total=len(df_sxul), file=sys.stdout, unit=" symexpiry") as tqr:
        for i in df_sxul.itertuples():
            tqr.set_description(f"NSE option chains {str(i.symbol+':'+i.expiry).ljust(22)}")
            chains.append(catch(lambda: get_nse_chain(i.symbol, i.expiry, i.undPrice, i.lot)))
            tqr.update(1)

    # remove empty elements in list of dfs and concatenate
    df_chains = pd.concat([x for x in chains if str(x) != 'nan'])

    # remove nan from prices
    df_chains = df_chains.dropna(subset=['cBid', 'cAsk', 'cLTP', 'pBid', 'pAsk', 'pLTP']).reset_index(drop=True)

    # convert symbols - friendly to IBKR
    df_chains = df_chains.assign(symbol=df_chains.symbol.str.slice(0,9))

    ntoi = {'M%26M': 'MM', 'M%26MFIN': 'MMFIN', 'L%26TFH': 'LTFH', 'NIFTY': 'NIFTY50', 'CHOLAFIN':'CIFC'}
    df_chains.symbol = df_chains.symbol.replace(ntoi)

    # set the types for indexes as IND
    ix_symbols = ['NIFTY50', 'BANKNIFTY', 'NIFTYIT']

    # build the underlying contracts
    scrips = list(df_chains.symbol.unique())
    und_contracts = [Index(symbol=s, exchange=exchange) if s in ix_symbols else Stock(symbol=s, exchange=exchange) for s in scrips]

    # get the underlying conIds
    with get_connected('nse', 'live') as ib:
        qual_unds = ib.qualifyContracts(*und_contracts)

    df_chains = df_chains.assign(undId = df_chains.symbol.map({q.symbol: int(q.conId) for q in qual_unds}))

    # convert datetime to correct format for IB
    df_chains = df_chains.assign(expiry=[e.strftime('%Y%m%d') for e in df_chains.expiry])
                                
    return df_chains

def tp_chains():
    '''Make df_chains from tradeplus
    Args: None
    Returns: dataframe of option chains'''

    # extract from tradeplusonline
    tp = pd.read_html('https://www.tradeplusonline.com/Equity-Futures-Margin-Calculator.aspx')
    df_tp = tp[1][2:].iloc[:, :3].reset_index(drop=True)
    df_tp.columns=['symbol', 'lot', 'undPrice']
    df_tp = df_tp.apply(pd.to_numeric, errors='ignore') # convert lot and undPrice to numeric

    # convert symbols - friendly to IBKR
    df_tp = df_tp.assign(symbol=df_tp.symbol.str.slice(0,9))
    ntoi = {'M&M': 'MM', 'M&MFIN': 'MMFIN', 'L&TFH': 'LTFH', 'NIFTY': 'NIFTY50'}
    df_tp.symbol = df_tp.symbol.replace(ntoi)

    # set the types for indexes as IND
    ix_symbols = ['NIFTY50', 'BANKNIFTY', 'NIFTYIT']

    # build the underlying contracts
    scrips = list(df_tp.symbol)
    und_contracts = [Index(symbol=s, exchange=exchange) if s in ix_symbols else Stock(symbol=s, exchange=exchange) for s in scrips]

    # log to nse_chains.log
    with open(logpath+'nse_chains.log', 'w'):
        pass # clear the run log
    util.logToFile(logpath+'nse_chains.log')

    # build the chains
    with get_connected('nse', 'live') as ib:
        contracts=ib.qualifyContracts(*und_contracts)
        chains = {}
        with tqdm(total=len(contracts), file=sys.stdout, unit=' contract') as tqc:
            for contract in contracts:
                tqc.set_description(f"Getting strikes & expiries for {contract.symbol.ljust(9)}")
                chains[contract.symbol] = catch(lambda: ib.reqSecDefOptParams(underlyingSymbol=contract.symbol, futFopExchange='', 
                                              underlyingConId=contract.conId, underlyingSecType=contract.secType))
                tqc.update(1)

    # build the chain dataframe
    sek = [(product([k], v.expirations, v.strikes, [v.underlyingConId])) for k, m in chains.items() for v in m]
    df_chains = pd.DataFrame([i for s in sek for i in s], columns=['symbol', 'expiry', 'strike', 'undId'])

    df_chains = df_chains.set_index('symbol').join(df_tp.set_index('symbol')).reset_index()
    
    return df_chains

def get_chains(nseweb=True):
    '''Gets NSE chains.
    First preference given to nse web
    If error, gets it from tradeplus (2 mins)
    Arg: None
    Returns: df_chains'''
    if nseweb:
        try:
            df_chains = nse_chains()
        except Exception as e:
            df_chains = tp_chains()
    else:
        df_chains = tp_chains()
                                
    df_chains.to_pickle(fspath+'nse_chains.pkl') # write to pickle for size_chains to pickup
        
    return df_chains

#_____________________________________

