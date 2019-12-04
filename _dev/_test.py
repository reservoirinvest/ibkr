from ib_insync import IB, Stock, util
import asyncio
import json

import nest_asyncio
nest_asyncio.apply()

market = 'snp'

# ...variables initialization
with open('var.json', 'r') as fp:
    data = json.load(fp)

host = data['common']['host']
port = data[market]['port']
cid = 1

symbols = ['CELG', 'INTC']


async def get_dte(dt):
    return await (util.parseIBDatetime(dt) -
                  datetime.datetime.now().date()).days


async def basedata(ib, symbols):

    contracts = [await ib.qualifyContractsAsync(Stock(s, 'SMART', 'USD')) for s in symbols]

    # Dictionary to eliminate duplicate symbols
    contracts = {s.symbol: s for c in contracts for s in c}

    d = {}

    # get prices, chains and ohlcs with standard deviaPtion
    for contract in contracts.values():
        tick = await ib.reqTickersAsync(contract)
        chain = await ib.reqSecDefOptParamsAsync(
            underlyingSymbol=contract.symbol, futFopExchange='',
            underlyingSecType=contract.secType, underlyingConId=contract.conId)

        ohlc = await ib.reqHistoricalDataAsync(contract=contract, endDateTime='',
                                               durationStr='365 D', barSizeSetting='1 day',
                                               whatToShow='Trades', useRTH=True)
        ohlc = util.df(ohlc)

        ohlc = ohlc.assign(dte=ohlc.index.astype(int))

        ohlc = ohlc.assign(sd=ohlc.close.expanding(1).std(ddof=0))

        ohlc.insert(1, 'symbol', contract.symbol)
        # ohlc = ohlc.groupby('symbol').apply(lambda df: df.sort_values(
        #     'date', ascending=False)).reset_index(drop=True)
        # ohlcsd = ohlc.groupby('symbol').apply(
        #     lambda df: df.assign(sd=df.close.expanding(1).std(ddof=0)))

        d[contract.symbol] = (tick[0].marketPrice(),
                              chain[0].expirations, chain[0].strikes, ohlc)

    return d

with IB().connect(host=host, port=port, clientId=cid) as ib:
    ticksnchains = ib.run(basedata(ib, symbols))
