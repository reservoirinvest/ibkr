# x.py
from helper import *
from nse_func import *

# Do assignments
a = assign_var('snp')
for v in a:
    exec(v)

from ib_insync import *

ib =  get_connected('snp', 'live')

with open(logpath+'ztest.log', 'w'):
    pass # clear the run log

util.logToFile(logpath+'ztest.log')

#_____________________________________

# do_hist.py
def do_hist(ib, undId):
    '''Historize ohlc
    Args:
        (ib) as connection object
        (undId) as contractId for underlying symbol in int
    Returns:
        df_hist as dataframe
        pickles the dataframe by symbol name
    '''
    qc = ib.qualifyContracts(Contract(conId=int(undId)))[0]
    hist = ib.reqHistoricalData(contract=qc, endDateTime='', 
                                        durationStr='365 D', barSizeSetting='1 day',  
                                                    whatToShow='Trades', useRTH=True)
    df_hist = util.df(hist)
    df_hist = df_hist.assign(date=pd.to_datetime(df_hist.date, format='%Y-%m-%d'))
    df_hist.insert(loc=0, column='symbol', value=qc.symbol)
    df_hist = df_hist.sort_values('date', ascending = False).reset_index(drop=True)
    df_hist.to_pickle(fspath+'_'+qc.symbol+'_ohlc.pkl')
    return None

#_____________________________________

