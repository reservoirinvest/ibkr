# x.py
from helper import *
from nse_func import *


# Do assignments
a = assign_var('nse')
for v in a:
    exec(v)

from ib_insync import *

# ib =  get_connected('nse', 'live')

with open(logpath+'ztest.log', 'w'):
    pass # clear the run log

util.logToFile(logpath+'ztest.log')

#_____________________________________

# get_lots.py
def get_lots():
    '''Get the lots with expiry dates
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

# get_xu_async.py
async def get_xu_async(symbol: str) -> pd.DataFrame():
    '''Gets the symbol, expiry, undPrice'''
    # get expiries for the symbol
    url = 'https://www.nseindia.com/live_market/dynaContent/live_watch/option_chain/optionKeys.jsp?symbol='
    xpd = "//*[@id='date']" # xpath for date select options
    xpu = "//*[@id='whttp://localhost:8888/notebooks/bin/z_nse_scrape_async.ipynb#rapper_btm']/table[1]/tr/td[2]/div/span[1]/b" # xpath for undPrice

    print(f'Started aiohttp for {symbol}')
    
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url+symbol) as resp:
            assert resp.status == 200
            res = await resp.text()
            print(f'Completed aiohttp for {symbol}')
            
            return res

async def df_when_done(tasks):
    from lxml import html
    dfs = []
    for res in limited_as_completed(tasks, 20):
        htree = await html.fromstring(res) #html is from lxml 
        expiries = [opt.text for e in htree.xpath(xpd) for opt in e if 'Select' not in opt.text.strip('')]
        undPrice = [float(e.text.split(' ')[1]) for e in htree.xpath(xpu)][0]

        # convert above to a DataFrame
        df = pd.DataFrame(list(product([symbol], expiries, [str(undPrice)])), 
                          columns=['symbol', 'expiry', 'undPrice'])
        df = df.apply(pd.to_numeric, errors = 'ignore')
        print('df done!')
        dfs.append(df)
    return dfs

import asyncio
from itertools import islice

def limited_as_completed(coros, limit):
    """
    Run the coroutines (or futures) supplied in the
    iterable coros, ensuring that there are at most
    limit coroutines running at any time.
    Return an iterator whose values, when waited for,
    are Future instances containing the results of
    the coroutines.
    Results may be provided in any order, as they
    become available.
    """
    futures = [
        asyncio.ensure_future(c)
        for c in islice(coros, 0, limit)
    ]
    async def first_to_finish():
        while True:
            await asyncio.sleep(0)
            for f in futures:
                if f.done():
                    futures.remove(f)
                    try:
                        newf = next(coros)
                        futures.append(
                            asyncio.ensure_future(newf))
                    except StopIteration as e:
                        pass
                    return f.result()
    while len(futures) > 0:
        yield first_to_finish()

#_____________________________________

