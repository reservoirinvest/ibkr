import pandas as pd
import numpy as np
import itertools
from ib_insync import Stock, Index, util, Order, Option

from helper import get_div_tick, get_dte, filter_kxdte, get_bsm
from ohlc import get_ohlc

# declarations

exchange = 'SMART'
currency = 'USD'
fspath = '../data/snp/'

maxdte = 70  # max expiry date for options
mindte = 20  # min expiry date for options
blks = 50
tradingdays = 252

#... make the rates
#__________________

rate_url = 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/textview.aspx?data=yield'
df_rate = pd.read_html(rate_url)[1]
df_rate.columns  = df_rate.iloc[0] # Set the first row as header
df_rate = df_rate.drop(0,0) # Drop the first row
rate = float(df_rate[-1:]['1 yr'].values[0])/100 # Get the last row's 1 yr value as float

#... build the symbols
#______________________

# ... build the snp list
sym_chg_dict = {'BRK.B': 'BRK B', 'BRK/B': 'BRK B'} # Remap symbols in line with IBKR

snpurl = 'https://en.wikipedia.org/wiki/S%26P_100'
df_snp = pd.read_html(snpurl, header=0)[2]

df_snp.Symbol = df_snp.Symbol.map(sym_chg_dict).fillna(df_snp.Symbol)
df_snp['Type'] = 'Stock'

# Download cboe weeklies to a dataframe
dls = "http://www.cboe.com/publish/weelkysmf/weeklysmf.xls"

# read from row no 11, dropna and reset index
df_cboe = pd.read_excel(dls, header=12, 
                        usecols=[0,2,3]).loc[11:, :]\
                        .dropna(axis=0)\
                        .reset_index(drop=True)

# remove column names white-spaces and remap to IBKR
df_cboe.columns = df_cboe.columns.str.replace(' ', '')
df_cboe.Ticker = df_cboe.Ticker.map(sym_chg_dict).fillna(df_cboe.Ticker)

# list the equities
equities = [e for e in list(df_snp.Symbol) if e in list(df_cboe.Ticker)]

# filter and list the etfs
df_etf = df_cboe[df_cboe.ProductType == 'ETF'].reset_index(drop=True)
etfs = list(df_etf.Ticker)

stocks = sorted(equities+etfs)

# list the indexes (sometimes does not work!!!)
# indexes = sorted('OEX,XEO,XSP'.split(','))
indexes = []

# Build a list of contracts
ss = [Stock(symbol=s, currency=currency, exchange=exchange) for s in set(stocks)]

# ixs = [Index(symbol=s,currency=currency, exchange='CBOE') for s in set(indexes)]
ixs = [Index(symbol=s,currency=currency, exchange='CBOE') for s in set(indexes)]

cs = ss+ixs

# sort in alphabetical order
cs.sort(key=lambda x: x.symbol, reverse=False)

def snp_list(ib):
    '''returns a list of qualified snp underlyings
    Args: ib as the active object
    Returns: qualified list with conID
    '''
    qcs = ib.qualifyContracts(*cs) # qualified underlyings
    
    #...following code remove scrips without option parameters
    # check for option chains

    qcs_chains = [{c.symbol: ib.reqSecDefOptParams(underlyingSymbol=c.symbol, futFopExchange='', 
                              underlyingConId=c.conId, underlyingSecType=c.secType)} for i in range(0, len(qcs), blks) for c in qcs[i: i+blks]]

    # remove symbols of those chains without any option parameters
    symbols = [k for q in qcs_chains for k, v in q.items() if v]

    # store chains to speeden up get_opt()
    symbols_chains = [v[0] for q in qcs_chains for k, v in q.items() if v]
    
    clean_qcs = [c for c in qcs if c.symbol in symbols]
    
    return (symbols, clean_qcs, symbols_chains)

def snp_und(ib, contract):
    '''returns the underlying details
    Args: 
       (ib) as the active ib object
       (contract) as the contract
    Returns: None. The underlying is pickled to _und.pkl'''

    ticker = get_div_tick(ib, contract)

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

    undlot = 100

    # margin of underlying
    order = Order(action='SELL', totalQuantity=undlot, orderType='MKT')

    margin = float(ib.whatIfOrder(contract, order).initMarginChange)

    df_und['margin'] = margin
    df_und['lot'] = undlot

    df_und.to_pickle(fspath+contract.symbol+'_und.pkl')
    
    return df_und

def get_opt(ib, contract, chain):
    '''returns the valid options and pickles them 
    Args: 
        (ib) as the active ib object 
        (contract) as the contract object 
        (chain) for the option chain with strikes and expiries 
    Returns: options dataframe''' 

    # get the underlying 

    df_und = snp_und(ib, contract)
    divrate = df_und.divrate.item() # extract the dividend rate 
    
    # get the ohlc 
    df_ohlc = get_ohlc(ib, contract, fspath) 

    # symbol
    symbol = contract.symbol

    # rights
    right = ['P', 'C'] 

    undPrice = df_und.undPrice[0]

    # limit the expirations to between min and max dates
    expirations = sorted([exp for exp in chain.expirations 
                          if mindte < get_dte(exp) < maxdte])

    # get the strikes
    strikes = sorted([strike for strike in chain.strikes])

    # build the puts and calls
    rights = ['P', 'C']

    df_tgt = pd.DataFrame([i for i in itertools.product([symbol], expirations, strikes, rights)], columns=['symbol', 'expiry', 'strike', 'right'])
    df_tgt['dte'] = [get_dte(e) for e in df_tgt.expiry]

    df_tgt1 = filter_kxdte(df_tgt, df_ohlc)

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

    df_und1 = df_und[['symbol', 'undPrice', 'lot', 'margin']].set_index('symbol') # get respective columns from df_und

    df_opt = df_opt.set_index('symbol').join(df_und1) # join for lot and margin

    # get the standard deviation based on days to expiry
    df_opt = df_opt.assign(stdev=[df_ohlc.iloc[i].stdev for i in df_opt.dte])

    # get the volatality based on days to expiry
    df_opt = df_opt.assign(volatility=[df_ohlc.iloc[i].volatility for i in df_opt.dte])

    # high52 and low52 for the underlying
    df_opt = df_opt.assign(hi52 = df_ohlc[:252].high.max())
    df_opt = df_opt.assign(lo52 = df_ohlc[:252].low.min())
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

    tickers = [ib.reqTickers(*cs[i: i+100]) for i in range(0, len(cs), 100)]

    df_opt = df_opt.assign(price=[t.marketPrice() for ts in tickers for t in ts])

    df_opt = df_opt.assign(rom=df_opt.price/df_opt.margin*tradingdays/df_opt.dte*df_opt.lot)

    df_opt.to_pickle(fspath+contract.symbol+'_opt.pkl')
    
    return None