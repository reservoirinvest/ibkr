# ohlcs.py
"""Program that generates ohlcs
Date: 23-June-2019
Ver: 1.0
"""

from z_helper import *

#... check for nse_chains.pkl
if path.isfile(fspath+'nse_chains.pkl'):
    df_chains = pd.read_pickle(fspath+'nse_chains.pkl')
else:
    df_chains = nse_chains()

df_chains = df_chains[df_chains.symbol.isin(['BANKNIFTY', 'PNB'])]  # DATA LIMITER!!!

id_sym = df_chains.set_index('undId').symbol.to_dict()

#... get the ohlcs
ohlcs = []
with get_connected('nse', 'live') as ib:
    with tqdm(total= len(id_sym), file=sys.stdout, unit= 'symbol') as tqh:
        for k, v in id_sym.items():
            tqh.set_description(f"Getting OHLC hist frm IBKR for {v.ljust(9)}")
            ohlcs.append(catch(lambda:do_hist(ib,k)))
            tqh.update(1)

df_ohlcs = pd.concat(ohlcs)

# put stdev in ohlc
df_ohlcs = df_ohlcs.assign(stDev=df_ohlcs.groupby('symbol').close.transform(lambda x: x.expanding(1).std(ddof=0)))

#_____________________________________

