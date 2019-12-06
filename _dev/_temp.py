


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

        # ohlc = ohlc.assign(sd=ohlc.close.expanding(1).std(ddof=0))

        # ohlc.insert(1, 'symbol', contract.symbol)
        # ohlc = ohlc.groupby('symbol').apply(lambda df: df.sort_values(
        #     'date', ascending=False)).reset_index(drop=True)
        # ohlcsd = ohlc.groupby('symbol').apply(
        #     lambda df: df.assign(sd=df.close.expanding(1).std(ddof=0)))

        d[contract.symbol] = (tick[0].marketPrice(),
                              chain[0].expirations, chain[0].strikes, ohlc)

    return d

if __name__ == "__main__":

    with IB().connect(host=host, port=port, clientId=cid) as ib:
        symbols = list(ib.run(get_symbols()).symbol)
        symbols = ['CELG', 'INTC']
        ticksnchains = ib.run(basedata(ib, symbols))

print(ticksnchains)

# async def ctAsync(ib, symbol):
#     """ awaitable contract qualification for the symbol """

#     contract = Stock(symbol, 'SMART', 'USD')

#     return await ib.qualifyContractsAsync(contract)


# async def tickAsync(ib, contract):
#     """ awaitable price for the underlying contract """

#     return await ib.reqTickersAsync(contract)


# async def chainsAsync(ib, c):
#     """ awaitable chains for the underlying contract """

#     return await ib.reqSecDefOptParamsAsync(c.symbol, '', c.conId, c.secType)


# async def ticksChainsAysnc(ib, contracts):
#     """ awaitable ticks and chains for underlying contract """

#     task1 = [ib.reqTickersAsync(c) for c in contracts]
#     task2 = [ib.reqSecDefOptParamsAsync(underlyingSymbol=c.symbol, futFopExchange='',
#                                         underlyingConId=c.conId, underlyingSecType=c.secType) for c in contracts]
#     # Get a alternating list
#     tasks = [None]*(len(task1)+len(task2))
#     tasks[::2] = task1
#     tasks[1::2] = task2

#     return await asyncio.gather(*tasks)


# async def main():
#     """ bringing it all together """

#     market = 'snp'

#     # ...variables initialization
#     with open('var.json', 'r') as fp:
#         data = json.load(fp)

#     host = data['common']['host']
#     port = data[market]['port']
#     cid = 1

#     df_symbols = IB().run(snpsyms())

#     # !!! DATA LIMITER - 5 equities and 3 ETFs
#     df_symbols = pd.concat(
#         [df_symbols[df_symbols.ProductType == 'Equity'].head(7), df_symbols.head(3)])

#     symbols = list(df_symbols.symbol)

#     with IB().connect(host=host, port=port, clientId=cid) as ib:

#         # Get the contracts
#         contracts = [ib.run(ctAsync(ib, s)) for s in symbols]

#         # Distill out the ones with errors
#         contracts = [cs for c in contracts for cs in c if c]

#         # Get the ticks for the contracts
#         ticks = ib.run(ticksChainsAysnc(ib, contracts))

#         # tick = tickAsync(ib, contract)
#         # chain = chains (ib, contract)

#     return ticks

# # for non-imported modules
# if __name__ == '__main__':
#     print(f"started at {time.strftime('%X')}")
#     ticksnchains = IB().run(main())
#     print(f"finished at {time.strftime('%X')}")

#     # print(ticksnchains)