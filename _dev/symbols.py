import pandas as pd
import time
from ib_insync import IB, Stock
import asyncio

import json
import nest_asyncio
nest_asyncio.apply()


async def snpsyms():

    # Download cboe weeklies
    dls = "http://www.cboe.com/publish/weelkysmf/weeklysmf.xls"

    # read from row no 11, dropna and reset index
    df_cboe = pd.read_excel(dls, header=12,
                            usecols=[0, 2, 3]).loc[11:, :]\
        .dropna(axis=0)\
        .reset_index(drop=True)

    # remove column names white-spaces and remap to IBKR
    df_cboe.columns = df_cboe.columns.str.replace(' ', '')

    # remove '/' for IBKR
    df_cboe.Ticker = df_cboe.Ticker.str.replace('/', ' ', regex=False)

    snp100 = list(pd.read_html('https://en.wikipedia.org/wiki/S%26P_100',
                               header=0, match='Symbol')[0].loc[:, 'Symbol'])
    # without dot in symbol
    snp100 = [s.replace('.', ' ') if '.' in s else s for s in snp100]

    # remove equities not in snp100
    df_symbols = df_cboe[~((df_cboe.ProductType == 'Equity')
                           & ~df_cboe.Ticker.isin(snp100))]

    # rename Ticker to symbols
    df_symbols = df_symbols.rename({'Ticker': 'symbol'}, axis=1)

    # add in the lots
    df_symbols = df_symbols.assign(lot=100)

    return df_symbols


async def ctAsync(ib, symbol):
    """ awaitable contract qualification for the symbol """

    contract = Stock(symbol, 'SMART', 'USD')

    return await ib.qualifyContractsAsync(contract)


async def tickAsync(ib, contract):
    """ awaitable price for the underlying contract """

    return await ib.reqTickersAsync(contract)


async def chainsAsync(ib, c):
    """ awaitable chains for the underlying contract """

    return await ib.reqSecDefOptParamsAsync(c.symbol, '', c.conId, c.secType)


async def ticksChainsAysnc(ib, contracts):
    """ awaitable ticks and chains for underlying contract """

    task1 = [ib.reqTickersAsync(c) for c in contracts]
    task2 = [ib.reqSecDefOptParamsAsync(underlyingSymbol=c.symbol, futFopExchange='',
                                        underlyingConId=c.conId, underlyingSecType=c.secType) for c in contracts]
    # Get a alternating list
    tasks = [None]*(len(task1)+len(task2))
    tasks[::2] = task1
    tasks[1::2] = task2

    return await asyncio.gather(*tasks)


async def main():
    """ bringing it all together """

    market = 'snp'

    # ...variables initialization
    with open('var.json', 'r') as fp:
        data = json.load(fp)

    host = data['common']['host']
    port = data[market]['port']
    cid = 1

    df_symbols = IB().run(snpsyms())

    # !!! DATA LIMITER - 5 equities and 3 ETFs
    df_symbols = pd.concat(
        [df_symbols[df_symbols.ProductType == 'Equity'].head(7), df_symbols.head(3)])

    symbols = list(df_symbols.symbol)

    with IB().connect(host=host, port=port, clientId=cid) as ib:

        # Get the contracts
        contracts = [ib.run(ctAsync(ib, s)) for s in symbols]

        # Distill out the ones with errors
        contracts = [cs for c in contracts for cs in c if c]

        # Get the ticks for the contracts
        ticks = ib.run(ticksChainsAysnc(ib, contracts))

        # tick = tickAsync(ib, contract)
        # chain = chains (ib, contract)

    return ticks

# for non-imported modules
if __name__ == '__main__':
    print(f"started at {time.strftime('%X')}")
    ticksnchains = IB().run(main())
    print(f"finished at {time.strftime('%X')}")

    # print(ticksnchains)
