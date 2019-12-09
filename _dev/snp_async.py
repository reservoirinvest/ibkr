import pandas as pd
import numpy as np
import datetime
import json
from itertools import product

import time
from math import erf, sqrt
from ib_insync import IB, Stock, Option, util

import asyncio

import nest_asyncio
nest_asyncio.apply()

market = "snp"

# ...variables initialization
with open("var.json", "r") as fp:
    data = json.load(fp)

host = data["common"]["host"]
port = data[market]["port"]
cid = 1


async def get_symbols():

    # Download cboe weeklies
    dls = "http://www.cboe.com/publish/weelkysmf/weeklysmf.xls"

    # read from row no 11, dropna and reset index
    df_cboe = (
        pd.read_excel(dls, header=12, usecols=[0, 2, 3])
        .loc[11:, :]
        .dropna(axis=0)
        .reset_index(drop=True)
    )

    # remove column names white-spaces and remap to IBKR
    df_cboe.columns = df_cboe.columns.str.replace(" ", "")

    # remove '/' for IBKR
    df_cboe.Ticker = df_cboe.Ticker.str.replace("/", " ", regex=False)

    snp100 = list(
        pd.read_html(
            "https://en.wikipedia.org/wiki/S%26P_100", header=0, match="Symbol"
        )[0].loc[:, "Symbol"]
    )
    # without dot in symbol
    snp100 = [s.replace(".", " ") if "." in s else s for s in snp100]

    # remove equities not in snp100
    df_symbols = df_cboe[
        ~((df_cboe.ProductType == "Equity") & ~df_cboe.Ticker.isin(snp100))
    ]

    # rename Ticker to symbols
    df_symbols = df_symbols.rename({"Ticker": "symbol"}, axis=1)

    return df_symbols

async def qual_contract(ib, contract)

async def get_contract(ib, symbol):
    return await ib.qualifyContractsAsync(Stock(symbol, "SMART", "USD"))

async def get_ohlc(ib, contract):
    ohlc = await ib.reqHistoricalDataAsync(
        contract=contract,
        endDateTime="",
        durationStr="365 D",
        barSizeSetting="1 day",
        whatToShow="Trades",
        useRTH=True,
    )

    # reverse sort to have latest date on top
    df_ohlc = util.df(ohlc).sort_index(ascending=False).reset_index(drop=True)
    
    df_ohlc.insert(0, 'symbol', contract.symbol)

    df_ohlc["rise"] = [
        df_ohlc["close"].rolling(i).apply(lambda x: x[0] - x[-1], raw=True).max()
        for i in range(1, len(df_ohlc) + 1)
    ]
    df_ohlc["rise"] = df_ohlc.rise.abs()

    df_ohlc["fall"] = [
        df_ohlc["close"].rolling(i).apply(lambda x: x[0] - x[-1], raw=True).min()
        for i in range(1, len(df_ohlc) + 1)
    ]
    df_ohlc["fall"] = df_ohlc.fall.abs()

    df_ohlc = df_ohlc.assign(sd=df_ohlc.close.expanding(1).std(ddof=0))
    df_ohlc.sd = df_ohlc.sd.expanding(1).max() # roll the standard deviation upwards

    return df_ohlc


async def get_tick(ib, contract):
    return await ib.reqTickersAsync(contract)


async def get_chain(ib, contract):
    return await ib.reqSecDefOptParamsAsync(
        underlyingSymbol=contract.symbol,
        futFopExchange="",
        underlyingSecType=contract.secType,
        underlyingConId=contract.conId,
    )

async def get_margin(ib, contract, order):
    return await ib.whatIfOrder(contract, order)

if __name__ == "__main__":
    
    start = datetime.datetime.now()
    print(f"started at {time.strftime('%X')}")

    with IB().connect(host=host, port=port, clientId=cid) as ib:
        symbols = ib.run(get_symbols()).symbol
        symbols = ["CELG", "GOOG"]

        contracts = [ib.run(get_contract(ib, symbol)) for symbol in symbols]
        contracts = [c for cs in contracts for c in cs if c]
        
        for c in contracts:
            
            df_ohlc = ib.run(get_ohlc(ib, c))

            undPrice = ib.run(get_tick(ib, c))[0].marketPrice()

            chains = {c.symbol: ib.run(get_chain(ib, c))[0]}

            sek = {b for a in [list(product([k], m.expirations, m.strikes))
                            for k, m in chains.items()] for b in a}

            dfc = pd.DataFrame(list(sek), columns=['symbol', 'expiry', 'strike'])
            dfc = dfc.assign(dte=[(util.parseIBDatetime(
                dt)-datetime.datetime.now().date()).days for dt in dfc.expiry])    
            dfc = dfc[dfc.dte <= data['common']['maxdte']] # Limit to max and min dte
            dfc = dfc[dfc.dte >= data['common']['mindte']]
            dfc = dfc.join(dfc.dte.apply(lambda x: df_ohlc.iloc[x][['rise', 'fall', 'sd']])) # integrate rise, fall and stdev

            # remove the calls and puts whose strike is in the threshold of st dev
            dfc['undPrice'] = undPrice
            dfc = dfc.assign(right=np.where(dfc.strike >= dfc.undPrice, 'C', 'P'))
            c_mask = (dfc.right == 'C') & (dfc.strike > dfc.undPrice + data['common']['callstdmult']*dfc.sd)
            p_mask = (dfc.right == 'P') & (dfc.strike < dfc.undPrice - data['common']['putstdmult']*dfc.sd)
            dfc = dfc[c_mask | p_mask].reset_index(drop=True)

            # Based on filter selection in json weed out...
            dfc = dfc.assign(strikeRef = np.where(dfc.right == 'P', 
                                                dfc.undPrice-dfc.fall, 
                                                dfc.undPrice+dfc.rise))

            if data['common']['callRise']:
                dfc = dfc[~((dfc.right == 'C') & (dfc.strike < dfc.strikeRef))].reset_index(drop=True)

            if data['common']['putFall']:
                dfc = dfc[~((dfc.right =='P') & (dfc.strike > dfc.strikeRef))].reset_index(drop=True)

            if data['common']['onlyPuts']:
                dfc = dfc[dfc.right == 'P'].reset_index(drop=True)

            # limit to nBands
            nBand = data['common']['nBand']
            gb = dfc.groupby(['right'])

            if 'C' in [k for k in gb.indices]:
                df_calls = gb.get_group('C').reset_index(drop=True).sort_values(['symbol', 'dte', 'strike'], ascending=[True, True, True])
                df_calls = df_calls.groupby(['symbol', 'dte']).head(nBand)
            else:
                df_calls = pd.DataFrame([])

            if 'P' in [k for k in gb.indices]:
                df_puts = gb.get_group('P').reset_index(drop=True).sort_values(['symbol', 'dte', 'strike'], ascending=[True, True, False])
                df_puts = df_puts.groupby(['symbol', 'dte']).head(nBand)
            else:
                df_puts =  pd.DataFrame([])

            dfc = pd.concat([df_puts, df_calls]).reset_index(drop=True)

            # qualify the options
            opts = [Option(i.symbol, i.expiry, i.strike, i.right, data[market]['exchange']) for i in dfc[['symbol', 'expiry', 'strike', 'right']].itertuples()]

    print(opts)
    
    end = datetime.datetime.now()-start
    print(f"finished at {time.strftime('%X')} in {end.microseconds*1e-6:.2f} seconds")

