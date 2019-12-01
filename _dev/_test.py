from ib_insync import IB, Stock
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

symbols = ['CELG', 'INTC', 'QQQ', 'CSCO']


async def basedata(ib, symbols):

    contracts = [await ib.qualifyContractsAsync(Stock(s, 'SMART', 'USD')) for s in symbols]

    # Dictionary to eliminate duplicate symbols
    contracts = {s.symbol: s for c in contracts for s in c}

    d = {}

    for contract in contracts.values():
        tick = await ib.reqTickersAsync(contract)
        chain = await ib.reqSecDefOptParamsAsync(
            underlyingSymbol=contract.symbol, futFopExchange='',
            underlyingSecType=contract.secType, underlyingConId=contract.conId)
        ohlc = await ib.reqHistoricalDataAsync(contract=contract, endDateTime='',
                                               durationStr='365 D', barSizeSetting='1 day',
                                               whatToShow='Trades', useRTH=True)

        d[contract.symbol] = (tick[0].marketPrice(),
                              chain[0].expirations, chain[0].strikes, ohlc)

    return d

async def stdevfr(ohlc, mindte, maxdte, callstdmult, putstdmult):
    """Std deviation, fall and rise w.r.t. dte for a symbol"""

    ohlcs = [i for j in ohlcs for i in j]

    # make the ohlc dataframe
    df_ohlc = pd.DataFrame()
    for i, o in enumerate(ohlcs):
        df = util.df(o)
        if not o:
            print(f'{und_contracts[i].symbol} ohlc is empty')
        else:
            df_ohlc = df_ohlc.append(df.assign(symbol=und_contracts[i].symbol))

    #... compute the standard deviations
    df_ohlc = df_ohlc.assign(date=pd.to_datetime(df_ohlc.date, format='%Y-%m-%d'))

    grp1 = df_ohlc.groupby('symbol')
    grp2 = grp1.apply(lambda df: df.sort_values('date', ascending=False)).reset_index(drop=True)

    df_ohlcsd = grp2.groupby('symbol').apply(lambda df: df.assign(stDev=df.close.expanding(1).std(ddof=0))).reset_index(drop=True)

    return df_ohlcsd

with IB().connect(host=host, port=port, clientId=cid) as ib:
    ticksnchains = ib.run(basedata(ib, symbols))
