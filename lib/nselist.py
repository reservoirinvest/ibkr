import pandas as pd
import numpy as np
from itertools import product, repeat
import datetime

from ib_insync import Stock, Index, util, Order, Option
from helper import get_div_tick, get_dte, filter_kxdte, get_bsm, catch
from ohlc import get_ohlc

# declarations

exchange = 'NSE'
fspath = '../data/nse/'
maxdte = 70  # max expiry date for options
mindte = 3  # min expiry date for options
blks = 50
tradingdays = 252

#... make the rates
#__________________

#... Get risk-free rate

rate_url = pd.read_html('https://www.fbil.org.in/')[0]

rate = float(rate_url[3][1])/100

#... build the symbols
#______________________

tp = pd.read_html('https://www.tradeplusonline.com/Equity-Futures-Margin-Calculator.aspx')

df_tp = tp[1][2:].iloc[:, :-1]
df_tp = df_tp.iloc[:, [0,1,5]]
df_tp.columns=['nseSymbol', 'lot', 'margin']

cols = df_tp.columns.drop('nseSymbol')
df_tp[cols] = df_tp[cols].apply(pd.to_numeric, errors='coerce') # convert lot and margin to numeric

df_slm = df_tp.copy()

# Truncate to 9 characters for ibSymbol
df_slm['ibSymbol'] = df_slm.nseSymbol.str.slice(0,9)

# nseSymbol to ibSymbol dictionary for conversion
ntoi = {'M&M': 'MM', 'M&MFIN': 'MMFIN', 'L&TFH': 'LTFH', 'NIFTY': 'NIFTY50'}

# remap ibSymbol, based on the dictionary
df_slm.ibSymbol = df_slm.ibSymbol.replace(ntoi)

# separate indexes and equities, eliminate discards from df_slm
indexes = ['NIFTY50', 'BANKNIFTY']
discards = ['NIFTYMID5', 'NIFTYIT', 'LUPIN']
equities = sorted([s for s in df_slm.ibSymbol if s not in indexes+discards])

symbols = equities+indexes

cs = [Stock(s, exchange) if s in equities else Index(s, exchange) for s in symbols]

# cs = cs[:5] # DATA LIMITER!!!

def nse_list(ib):
    '''returns a list of qualified nse underlyings, with symbols and option chains for them
    Args: ib as the active ib object
    Returns: qualified list with conID
    '''
    qcs = ib.qualifyContracts(*cs) # qualified underlyings

    qcs_chains = []
    for i in range(0, len(qcs), 50):
        for c in qcs[i: i+50]:
            qcs_chains.append(ib.reqSecDefOptParams(underlyingSymbol=c.symbol, futFopExchange='', 
                                  underlyingConId=c.conId, underlyingSecType=c.secType))
            ib.sleep(0.5)

    syms = [q.symbol for q in qcs]
    expirations = [catch(lambda: c.expirations) for q in qcs_chains for c in q]
    strikes = [catch(lambda: c.strikes) for q in qcs_chains for c in q]

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

def nse_und(ib, contract):
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

    #... get the lot, margin, undPrice and dividend rate
    undlot = df_slm.loc[df_slm.ibSymbol == contract.symbol, 'lot'].item()
    df_und['lot'] = undlot

    # margin of underlying
    order = Order(action='SELL', totalQuantity=undlot, orderType='MKT')

#     margin = float(ib.whatIfOrder(contract, order).initMarginChange) # doesn't work because permission is not there for NRIs!!!
    margin = df_slm.loc[df_slm.ibSymbol == contract.symbol, 'margin'].item()

    df_und['margin'] = margin

    df_und.to_pickle(fspath+contract.symbol+'_und.pkl')
    
    return df_und

def get_opt(ib, df):
    '''returns the valid options and pickles them 
    Args:
        (ib) as the active ib object
        (df) datframe with columns und_contract, expiry, strike
    Returns: options dataframe''' 

    und_contract = df.iloc[0].und_contract

    # get the underlying
    df_und = nse_und(ib, und_contract)
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