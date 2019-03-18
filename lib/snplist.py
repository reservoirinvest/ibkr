import pandas as pd
import numpy as np
from itertools import product, repeat
from ib_insync import Stock, Index, util, Order, Option
import datetime
from collections import OrderedDict

from helper import get_div_tick, get_dte, filter_kxdte, get_bsm, catch
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
sym_chg_dict = {'BRK.B': 'BRK B', 'BRK/B': 'BRK B', 'BRKB', 'BRK B'} # Remap symbols in line with IBKR

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

#...Build a list of contracts
stocks = [Stock(symbol=s, currency=currency, exchange=exchange) for s in set(stocks)]

# sort the stocks
s_dict = {s.symbol: s for s in stocks}
od = OrderedDict(sorted(s_dict.items()))
ss = [v for k, v in od.items()]

# ixs = [Index(symbol=s,currency=currency, exchange='CBOE') for s in set(indexes)]
ixs = [Index(symbol=s,currency=currency, exchange='CBOE') for s in set(indexes)]

cs = ss+ixs

# cs = cs[:5] # DATA LIMITER!!!

# sort in alphabetical order
cs.sort(key=lambda x: x.symbol, reverse=False)

def snp_list(ib):
    '''returns a list of qualified snp underlyings
    Args: ib as the active object
    Returns: qualified list with conID
    '''
    
    qcs = ib.qualifyContracts(*cs) # qualified underlyings

    qcs_chains = []
    for i in range(0, len(qcs), 50):
        for c in qcs[i: i+50]:
            qcs_chains.append(ib.reqSecDefOptParams(underlyingSymbol=c.symbol, futFopExchange='', 
                                  underlyingConId=c.conId, underlyingSecType=c.secType))
            ib.sleep(0.5)

    # remove chains which are not SMART
    qcs_smart = [c for q in qcs_chains for c in q if c.exchange == exchange]

    syms = [q.tradingClass for q in qcs_smart]
    expirations = [catch(lambda: q.expirations) for q in qcs_smart]
    strikes = [catch(lambda: q.strikes) for q in qcs_smart]

    df_symconchain = pd.DataFrame([{'symbol': s, 'expiry': e, 'strike': k, 'und_contract': q} 
                                   for s, e, k, q in zip(syms, expirations, strikes, qcs)]).dropna(how = 'any') # make dataframe

    scc = [s for i in [list(product(r[1][0], r[1][1], r[1][[2]])) 
                       for r in df_symconchain.T.items()] for s in i] # dataframe in a list of tuples

    df_scc = pd.DataFrame(scc, columns= ['expiry', 'strike', 'symbol']).merge(df_symconchain[['symbol', 'und_contract']], on='symbol') # integrate with und_contract

    # get the dte
    df_scc['dte'] = (pd.to_datetime(df_scc.expiry) - datetime.datetime.now()).dt.days

    # drop chains whose dte is less than mindte and more than maxdte
    df_scc1 = df_scc[df_scc.dte.between(mindte, maxdte)].drop('dte', 1)

    return df_scc1

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

def get_opt(ib, df):
    '''returns the valid options and pickles them 
    Args:
        (ib) as the active ib object
        (df) datframe with columns und_contract, expiry, strike
    Returns: options dataframe'''
    
    df = df.reset_index(drop=True)  # reset the index

    und_contract = df.iloc[0].und_contract
    
    # get the underlying
    df_und = snp_und(ib, und_contract)
    divrate = df_und.divrate.item() # extract the dividend rate 
    
    # get the ohlc 
    df_ohlc = get_ohlc(ib, und_contract, fspath) 

    # symbol
    symbol = und_contract.symbol
    
    undPrice = df_und.undPrice[0]
    
    # build the puts and calls
    df['right'] = np.where(df.strike < undPrice, 'P', 'C')

    df['dte'] = [get_dte(e) for e in df.expiry]

    df_tgt = filter_kxdte(df, df_ohlc)    
    
    # make the und_contracts
    und_contracts = [Option(symbol, expiry, strike, right, exchange) 
                 for symbol, expiry, strike, right 
                 in zip(df_tgt.symbol, df_tgt.expiry, df_tgt.strike, df_tgt.right)]

    qc = [ib.qualifyContracts(*und_contracts[i: i+blks]) for i in range(0, len(und_contracts), blks)]    

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
            zip(repeat(undPrice), df_opt.strike, df_opt.dte, repeat(rate), df_opt.volatility, repeat(divrate))]

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

    df_opt.to_pickle(fspath+und_contract.symbol+'_opt.pkl')
    
    return None