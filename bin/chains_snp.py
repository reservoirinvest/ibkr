# chains_snp.py
"""Program generates symbols and lots for SNP (USA)
Date: 25-June-2019
Ver: 1.0
"""

from z_helper import *

# from json
a = assign_var('snp')
for v in a:
    exec(v)

#...snp specific functions
#..........................

def get_chains(ib):
    '''Gets the symbols and lots from snp
    Arg: (ib) as connection object
    Returns: df_chains as dataframe of option chains'''
    
    # exclusion list (excludes symbols with existing positions!)
    excl = ['VXX','P', 'TSRO', 'GOOGL']

    # Download cboe weeklies to a dataframe
    dls = "http://www.cboe.com/publish/weelkysmf/weeklysmf.xls"

    snp100 = list(pd.read_html('https://en.wikipedia.org/wiki/S%26P_100', 
                               header=0, match='Symbol')[0].loc[:, 'Symbol'])
    snp100 = [s.replace('.', ' ') if '.' in s else s  for s in snp100] # without dot in symbol

    # read from row no 11, dropna and reset index
    df_cboe = pd.read_excel(dls, header=12, 
                            usecols=[0,2,3]).loc[11:, :]\
                            .dropna(axis=0)\
                            .reset_index(drop=True)

    # remove column names white-spaces and remap to IBKR
    df_cboe.columns = df_cboe.columns.str.replace(' ', '')

    # remove '/' for IBKR
    df_cboe.Ticker = df_cboe.Ticker.str.replace('/', ' ', regex=False)

    # make symbols
    symbols = {s for s in df_cboe.Ticker if s not in excl if s in snp100}

    df_symlot = pd.DataFrame({'symbol': sorted(list(symbols)),
                'lot': pd.Series(100, index=range(len(symbols)), dtype='int32')})
    und_contracts = [Stock(symbol=s, exchange=exchange, currency='USD') for s in symbols]

    # log to snp_chains.log
    with open(logpath+'snp_chains.log', 'w'):
        pass # clear the run log
    util.logToFile(logpath+'snp_chains.log')

    # build the chains
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

    df_chains = df_chains.set_index('symbol').join(df_symlot.set_index('symbol')).drop_duplicates().reset_index()
    
    df_chains.to_pickle(fspath+'snp_chains.pkl') # write to pickle for size_chains to pickup
    
    return df_chains

#_____________________________________

