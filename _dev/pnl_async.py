from ib_insync import IB
import asyncio
import json

import nest_asyncio
nest_asyncio.apply()


async def pnlAsync(ib):
    """ Async pnl
    Returns: PnL and funds as json"""

    acct = ib.managedAccounts()[0]

    accsum = ib.accountSummary(account=acct)

    pnlobj = ib.reqPnL(acct)

    await ib.pnlEvent

    fundDict = {t.tag: float(t.value)
                for t in accsum
                if t.tag in ["NetLiquidation", "AvailableFunds"]}

    pnlDict = {'dailyPnL': pnlobj.dailyPnL,
               'unrealizedPnL': pnlobj.unrealizedPnL,
               'realizedPnL': pnlobj.realizedPnL}

    return {**pnlDict, **fundDict}


async def get_pnlAsync(market):
    """ Gets the pnl and liquidity for the market """

    # ...variables initialization
    with open('var.json', 'r') as fp:
        data = json.load(fp)

    host = data['common']['host']
    port = data[market]['port']
    cid = 1

    # ...generate PnL with funds
    with await IB().connectAsync(host=host, port=port, clientId=cid) as ib:
        p = ib.run(pnlAsync(ib))

    return(p)

if __name__ == '__main__':

    output = IB().run(get_pnlAsync('snp'))

    print(output)
