# x.py

# ...imports
import pandas as pd
import numpy as np
import requests
import calendar
import datetime
import sys
import json

from ib_insync import *
from io import StringIO
from itertools import product
from os import listdir
from tqdm import tqdm
from math import floor, log10, ceil
from lxml import html

#_____________________________________

# nse_scrape.py

#...common helper functions
#..........................

# assign_var.py
def assign_var(market):
    '''Assign variables using exec
    Arg: (market) as string <'nse'>|<'snp' 
    Returns: VarList as a list of strings containing assignments
             These will be executed upon using exec()'''

    with open('variables.json', 'r') as fp:
        varDict = json.load(fp)
    
    varList = [str(k+"='"+str(v)+"'")  if type(v) is str
               else (str(k+'='+ str(v))) if type(v) is list
               else str(k+'='+str(v)) for k, v in varDict[market].items()]
    return varList

# get_prec.py
# get precision, based on the base
def get_prec(v, base):
    '''gives the precision value
    args:
       (v) as value needing precision in float
       (base) as the base value e.g. 0.05'''
    
    return round(round((v)/ base) * base, -int(floor(log10(base))))

# get_connected.py
def get_connected(market, trade_type):
    ''' get connected to ibkr
    Args: 
       (market) as string <'nse'> | <'snp'>
       (trade_type) as string <'live'> | <'paper'>
    Returns:
        (ib) object if successful
    '''
    
    ip = (market.upper(), trade_type.upper())
    
    #host dictionary
    hostdict = {('NSE', 'LIVE'): 3000,
                ('NSE', 'PAPER'): 3001,
                ('SNP', 'LIVE'): 1300,
                ('SNP', 'PAPER'): 1301}
    
    host = hostdict[ip]
    
    cid = 1 # initialize clientId
    max_cid = 5 # maximum clientId allowed. max possible is 32

    for i in range(cid, max_cid):
        try:
            ib = IB().connect('127.0.0.1', host, clientId=i)
            
        except Exception as e:
            print(e) # print the error
            continue # go to next
            
        break # successful try
        
    return ib

# catch.py
def catch(func, handle=lambda e : e, *args, **kwargs):
    '''List comprehension error catcher
    Args: 
        (func) as the function
         (handle) as the lambda of function
         <*args | *kwargs> as arguments to the functions
    Outputs:
        output of the function | <np.nan> on error
    Usage:
        eggs = [1,3,0,3,2]
        [catch(lambda: 1/egg) for egg in eggs]'''
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return np.nan

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

# get_dte.py
def get_dte(dt):
    '''Gets days to expiry
    Arg: (dt) as day in string format 'yyyymmdd'
    Returns: days to expiry as int'''
    return (util.parseIBDatetime(dt) - 
            datetime.datetime.now().date()).days

# fallrise.py
def fallrise(df_hist, dte):
    '''Gets the fall and rise for a specific dte
    Args:
        (pklpath) as a string with ohlc pickle path
        (dte) as int for days to expiry
    Returns:
        (symbol, dte, fall, rise) as a tuple'''
#     df_hist = pd.read_pickle(pklpath)
    symbol = df_hist.symbol.unique()[0]
    df = df_hist.set_index('date').sort_index(ascending = True)
    df = df.assign(delta = df.high.rolling(dte).max() - df.low.rolling(dte).min(), 
                        pctchange = df.close.pct_change(periods=dte))

    df1 = df.sort_index(ascending = False)
    max_fall = df1[df1.pctchange<=0].delta.max()
    max_rise = df1[df1.pctchange>0].delta.max()
    
    return (symbol, dte, max_fall, max_rise)
    
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
    '''scrapes option chain from nse website pages'''
    
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

# remqty_nse.py
def remqty_nse(ib):
    '''generates the remaining quantities dictionary
    Args:
        (ib) as connection object
        <df_opt> as dataframe of options. If not provided defaults to opts.pickle
    Returns:
        dfrq as a datframe with symbol and remaining quantities
        '''
    # from portfolio
    #...............

    p = util.df(ib.portfolio()) # portfolio table

    if p: # something is in the portfolio!
        # extract option contract info from portfolio table
        dfp = pd.concat([p, util.df([c for c in p.contract])[util.df([c for c in p.contract]).columns[:7]]], axis=1).iloc[:, 1:]
        dfp = dfp.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})

        # get the total position
        dfp1 = dfp.groupby('symbol').sum()['position']
    else:
        dfp1 = pd.DataFrame()

    # from options pickle
    #....................

    # get the options
    df_opt = pd.read_pickle(fspath + 'options.pkl')
    df_opt = df_opt.assign(und_remq=(nse_assignment_limit/(df_opt.lot*df_opt.undPrice)).astype('int')) # remaining quantities in entire nse

    # compute the remaining quantities
    df_opt1 = df_opt.groupby('symbol').first()[['lot', 'margin', 'und_remq']]

    df_opt2 = df_opt1.join(dfp1).fillna(0).astype('int')

    if dfp1.empty:
        df_opt2 = df_opt2.assign(remqty=df_opt2.und_remq)
    else:
        df_opt2 = df_opt2.assign(remqty=df_opt2.und_remq+(df_opt2.position/df_opt2.lot).astype('int'))

    dfrq = df_opt2[['remqty']]

    return dfrq

# upd.py
def upd(ib, df_tgt):
    '''Updates the nse options' price and margin
    Takes 3 mins for 450 options
    Args:
       (ib) as connection object
       (df_tgt) as the dataframe from targets.pickle
    Returns: DataFrame with updated option price and margin'''
    
    # get the contracts
    cs=[Contract(conId=c) for c in df_tgt.optId]

    blks = [cs[i: i+blk] for i in range(0, len(cs), blk)]
    cblks = [ib.qualifyContracts(*s) for s in blks]
    contracts = [z for x in cblks for z in x]

    # update prices
    tickers = ib.reqTickers(*contracts)

    x = []
    def getTickers(tickers):
        x.append(tickers)
    ib.pendingTickersEvent += getTickers

    optPrices = {t.contract.conId: t.marketPrice() for t in tickers} # {symbol: undPrice}

    df_tgt = df_tgt.assign(optPrice = df_tgt.optId.map(optPrices))

    # get the margins
    orders = [Order(action='SELL', orderType='MKT', totalQuantity=lot, whatIf=True) for lot in df_tgt.lot]

    mdict = {c.conId: ib.whatIfOrder(c, o).initMarginChange for c, o in zip(contracts, orders)} # making dictionary takes time!

    # Remove very large number in margin (1.7976931348623157e+308)
    mdict = {key: value for key, value in mdict.items() if float(value) < 1e7}

    # assign the margins
    df_tgt = df_tgt.assign(margin=df_tgt.optId.map(mdict).astype('float'))

    # calculate rom
    df_tgt = df_tgt.assign(rom=df_tgt.optPrice/df_tgt.margin*365/df_tgt.dte*df_tgt.lot)
    
    return df_tgt

#_____________________________________

# assignments.py

# for requests
headers = { 
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36', 
'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 
'Accept-Language' : 'en-US,en;q=0.5',
'Accept-Encoding' : 'gzip', 
'DNT' : '1', # Do Not Track Request Header 
'Connection' : 'close'
}

# from json
a = assign_var('nse')
for v in a:
    exec(v)


ib =  get_connected('nse', 'live')

with open(logpath+'nse_scrape.log', 'w'):
    pass # clear the run log

util.logToFile(logpath+'nse_scrape.log')

#_____________________________________

# chains.py
#... Make the chains
#...................

# get the symbols and lots
df_lots = get_lots()
symbols = sorted(list(df_lots.symbol.unique()))

# symbols = [s for s in symbols if s in ['NIFTY', 'PNB']] # DATA LIMITER!!!
# symbols = symbols[:5] # DATA LIMITER!!!

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

# convert symbols - friendly to IBKR and pickle the chains
df_chains = df_chains.assign(symbol=df_chains.symbol.str.slice(0,9))

ntoi = {'M%26M': 'MM', 'M%26MFIN': 'MMFIN', 'L%26TFH': 'LTFH', 'NIFTY': 'NIFTY50', 'CHOLAFIN':'CIFC'}
df_chains.symbol = df_chains.symbol.replace(ntoi)

# set the types for indexes as IND
ix_symbols = ['NIFTY50', 'BANKNIFTY', 'NIFTYIT']

# build the underlying contracts
scrips = list(df_chains.symbol.unique())
und_contracts = [Index(symbol=s, exchange=exchange) if s in ix_symbols else Stock(symbol=s, exchange=exchange) for s in scrips]

# get the underlying conIds
qual_unds = ib.qualifyContracts(*und_contracts)
df_chains = df_chains.assign(undId = df_chains.symbol.map({q.symbol: q.conId for q in qual_unds}))

# convert datetime to correct format for IB
df_chains = df_chains.assign(expiry=[e.strftime('%Y%m%d') for e in df_chains.expiry])

df_chains.to_pickle(fspath+'nse_chains.pkl')

#... Historize
#.............

# historize individual underlyings
hists = [] # initialize a list of underlying histories. Will be empty.
with tqdm(total=len(qual_unds), file=sys.stdout, unit=' symbol') as tqh:
    for q in qual_unds:
        tqh.set_description(f"Getting OHLC hist frm IBKR for {q.symbol.ljust(9)}")
        hists.append(catch(lambda: do_hist(ib, q.conId)))  # makes ohlc.pkl
        tqh.update(1)

# Capture the ohlc pickles
ohlc_pkls = [f for f in listdir(fspath) if (f[-8:] == 'ohlc.pkl')]
df_ohlcs = pd.concat([pd.read_pickle(fspath+o) for o in ohlc_pkls]).reset_index(drop=True)

# put stdev in ohlc and consolidate
df_ohlcs = df_ohlcs.assign(stDev=df_ohlcs.groupby('symbol').close.transform(lambda x: x.expanding(1).std(ddof=0)))
df_ohlcs.to_pickle(fspath+'ohlcs.pkl')

#... Remove chains not meeting put and call std filter

# get dte and remove those greater than maxdte
df_chains = df_chains.assign(dte=df_chains.expiry.apply(get_dte))                    
df_chains = df_chains[df_chains.dte <= maxdte]

# generate std dataframe
df = df_ohlcs[['symbol', 'stDev']]  # lookup dataframe
df = df.assign(dte=df.groupby('symbol').cumcount()) # get the cumulative count for location as dte
df.set_index(['symbol', 'dte'])

df1 = df_chains[['symbol', 'dte']]  # data to be looked at
df2 = df1.drop_duplicates()  # remove duplicates

df_std = df2.set_index(['symbol', 'dte']).join(df.set_index(['symbol', 'dte']))

# join to get std in chains
df_chainstd = df_chains.set_index(['symbol', 'dte']).join(df_std).reset_index()

# columns for puts and calls
putcols = [c for c in list(df_chainstd) if c[0] != 'c']
callcols = [p for p in list(df_chainstd) if p[0] != 'p']

# make puts and calls dataframe with std filter
df_puts = df_chainstd[df_chainstd.strike < (df_chainstd.undPrice-(df_chainstd.stDev*putstdmult))][putcols]
df_puts = df_puts.assign(right = 'P')

df_calls = df_chainstd[df_chainstd.strike > (df_chainstd.undPrice+(df_chainstd.stDev*callstdmult))][callcols]
df_calls = df_calls.assign(right = 'C')

# rename puts columns by removing 'p'
df_puts = df_puts.rename(columns={p: p[1:] for p in putcols if p[0] == 'p'})

# rename calls calumns by removing 'c'
df_calls = df_calls.rename(columns= {c: c[1:] for c in callcols if c[0] == 'c'})

df_opt = pd.concat([df_puts, df_calls], sort=False).reset_index(drop=True)

# ...Get the rom
#...............

# collect the option contracts
optcons = [Option(i.symbol, i.expiry, i.strike, i.right) for i in df_opt[['symbol', 'expiry', 'strike', 'right']].itertuples()]

# dice them into blocks
optblks = [optcons[i: i+blk] for i in range(0, len(optcons), blk)]

# qualify the contracts in blocks
cblks = [ib.qualifyContracts(*s) for s in optblks]

opts = [z for x in cblks for z in x]

# extract contractId for the options
df_conId = util.df(opts).iloc[:, 1:6]
df_conId = df_conId.rename(columns={'lastTradeDateOrContractMonth': 'expiry'})

df_conId = df_conId.rename(columns={'conId':'optId'})

# merge it with the opts dataframe
df_opt = pd.merge(df_opt, df_conId, how='left', on=['symbol', 'expiry', 'strike', 'right'])

idcon_d = {o.conId: o for o in opts} # conId:contract dictionary for the option contracts

#...prepare contract order tuples for all option contracts to get initmargin
allcos = [(idcon_d[i.optId], Order(action='SELL', orderType='MKT', totalQuantity=i.lot, whatIf=True)) for i in df_opt[['optId', 'lot']].itertuples()]

margins = {} # initialize margins dictionary
with tqdm(total=len(allcos), file=sys.stdout, unit=' symexpiry') as tqm:
    for q in allcos:
        tqm.set_description(f"IBKR margins for  {q[0].localSymbol.ljust(22)}")
        margins.update({q[0].conId: float(ib.whatIfOrder(*q).initMarginChange)})
        tqm.update(1)

df_opt = df_opt.assign(margin=df_opt.optId.map(margins))

# get lo52 and hi52
df_opt = df_opt.set_index('symbol').join(df_ohlcs.groupby('symbol')
                                         .close.agg(['min', 'max'])
                                         .rename(columns={'min': 'lo52', 'max': 'hi52'})).reset_index()


# get symbol and dte for risefall
tup4fr = [(df_ohlcs[df_ohlcs.symbol == s.symbol], s.dte) for s in df_opt[['symbol', 'dte']].drop_duplicates().itertuples()]

# get fall and rise
df_fr = pd.DataFrame([fallrise(*t) for t in tup4fr], columns=['symbol', 'dte', 'fall', 'rise'])

# merge with df_opt
df_opt = pd.merge(df_opt, df_fr, on=['symbol', 'dte'])

# pickle the options
df_opt.to_pickle(fspath+'options.pkl')

#_____________________________________

# targets.py
#...Screening strategies

# get remaining quantity
dfrq = remqty_nse(ib)

# remove options which are not in remaining quantities
df_tgt = df_opt.set_index('symbol').join(dfrq).reset_index()
df_tgt = df_tgt[df_tgt.remqty>0]

# extract the safest targets
mask = ((df_tgt.right == 'P') & ((df_tgt.undPrice - df_tgt.strike) > df_tgt.fall*frmult)) | \
       ((df_tgt.right == 'C') & ((df_tgt.strike - df_tgt.undPrice) > df_tgt.rise))
df_tgt = df_tgt[mask]

# update price and margins
df_tgt = upd(ib, df_tgt)

# get the expected price
expRom = ceil(df_tgt.rom.max() * 100.0)/100.0 # maximum expected rom
df_tgt = df_tgt.assign(expRom = expRom, 
                       expPrice = get_prec(df_tgt.optPrice*expRom/df_tgt.rom, 0.05)).sort_values('rom', ascending=False)

# fill the nan with ask price
df_tgt = df_tgt.assign(expPrice=df_tgt.expPrice.fillna(df_tgt.Ask))

# set the quantities
df_tgt = df_tgt.assign(qty=np.minimum(maxsellqty,df_tgt.remqty))

df_tgt.loc[df_tgt.expPrice < minOptPrice, "expPrice"] = minOptPrice # give the minimum option price

# pickle the opts and the targets
df_tgt.to_pickle(fspath+'targets.pkl')
df_tgt.to_excel(fspath+'targets.xlsx', sheet_name='Target', index=False, freeze_panes=(1,1))

#_____________________________________

