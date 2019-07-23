# x_4Capstocks.py

"""Program to prepare BUY trades for Capstocks
Creates a spreadsheet called capstocks.xlsx that can be sent to Senjith
Date: 23-July-2019
Ver: 1.0
Time taken: milliseconds
"""

from z_helper import *

# from json
a = assign_var('nse') + assign_var('common')
for v in a:
    exec(v)

def capstocks(cap_blacklist):
    '''prepares BUY trade spreadsheet for Capstocks - for YesBank+Kashi - NRE
    Args:
        (cap_blacklist) as list of scrips already held. From Xn googlespreadsheet
    Dependancies: ohlcs.pkl and targets.pkl
    Returns:
        (df) as DataFrame of capstocks trades'''
    
    # Blacklist from Google Xn
    cap_blacklist = {c[:9] for c in cap_blacklist}

    # get the ohlcs
    df_ohlcs = pd.read_pickle(fspath+'ohlcs.pkl')
    df_targets = pd.read_pickle(fspath+'targets.pkl')

    # remove blacklists from targets
    df_targets = df_targets[~df_targets.symbol.isin(cap_blacklist)]

    #... get 50% of max 1-day Standard Deviation

    # get max fall rise for 1-day Standard deviation
    df_ohlc = df_ohlcs.assign(OneDayVar = abs(df_ohlcs[['symbol', 'close']].groupby('symbol').agg('diff')))

    df_onedayvar = df_ohlc[['symbol', 'OneDayVar']].groupby('symbol').agg('max')

    df = df_targets.drop_duplicates('symbol').set_index('symbol').join(df_onedayvar).reset_index()

    cols = ['symbol', 'undId', 'lot', 'undPrice', 'OneDayVar', 'lo52', 'hi52']
    df1 = df[cols]

    df1 = df1.assign(expPrice3 = df1.undPrice-(df1.OneDayVar/3).apply(lambda x: get_prec(x, prec)), 
                     expPrice2 = df1.undPrice-(df1.OneDayVar/2).apply(lambda x: get_prec(x, prec)), 
                     expPrice1 = df1.undPrice-(df1.OneDayVar/1.5).apply(lambda x: get_prec(x, prec)))

    # get the deltas against hi52 and lo52
    df2 = df1.assign(lodelta=df1.undPrice/df1.lo52, hidelta=df1.hi52/df1.undPrice).sort_values('lodelta')

    # filter out target buys
    df3 = df2[(df2.lodelta < df2.lodelta.mean()) & (df2.hidelta > df2.hidelta.mean())]

    df4 = pd.melt(df3[['symbol', 'lot', 'undPrice', 'expPrice1', 'expPrice2', 'expPrice3']], 
                    id_vars=['symbol','lot', 'undPrice'],
                    value_vars=['expPrice1', 'expPrice2', 'expPrice3'],
                    value_name='LimitPrice',
                    var_name='variable')

    # Remove negative limitPrice
    df5 = df4[df4.LimitPrice > 0].reset_index(drop=True).sort_values('symbol')

    # Get the multiple of lots to be bought
    df6 = df5.assign(mult=df5.variable.str[-1:].astype('int32'))

    df7=df6.assign(qty=(df6.lot/df6.mult/2).astype('int').apply(lambda x: int(10 * round(float(x)/10))), trade='BUY')
    
    df7.loc[df7.lot <= 30, 'qty'] = 2 # Make the quantity to be 3 for very high margin symbols [EICHERMOT, PAGEIND]

    df7 = df7.sort_values(['symbol', 'LimitPrice'], ascending=[True, False])

    df7[['symbol', 'trade', 'LimitPrice', 'qty']].to_excel(fspath+'capstocks.xlsx', freeze_panes=(1,1), index=False)

    # make a watchlist to see Analyst Recommendations. Use only those which are BUY, OUTPERFORM.
    watch = [('DES', s, 'STK', 'NSE') for s in df7.symbol.unique()]
    util.df(watch).to_csv(fspath+'watch.csv', header=None, index=None)
    
    return df7

#_____________________________________

