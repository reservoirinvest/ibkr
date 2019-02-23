import pandas as pd
from ib_insync import Stock, Index

currency = 'USD'
exchange = 'SMART'

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

def snp_unds(ib):
    '''returns a list of qualified snp underlyings
    Args: ib as the active object
    Returns: qualified list with conID
    '''
    qcs = ib.qualifyContracts(*cs) # qualified underlyings
    return qcs