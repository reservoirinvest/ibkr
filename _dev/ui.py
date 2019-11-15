from ib_insync import IB, util
import time
import asyncio

host = '127.0.0.1'
nse = 3000
snp = 1300
cid = 0

ib = IB().connect(host, nse, 0)

async def pnlcoro(ib):
    '''Gets the pnl object'''
    acct = ib.managedAccounts()[0]
    pnl = ib.reqPnL(acct)

    await ib.pnlEvent

    return pnl

pnl = ib.run(pnlcoro(ib))
print(pnl)

print("\n*******\n")

accsum = ib.accountSummary(account=ib.managedAccounts()[0])
df_ac = util.df(accsum)
df = df_ac[df_ac.tag.isin(["NetLiquidation", "AvailableFunds"])]

print(df)

print("\n*******\n")