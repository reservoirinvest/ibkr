# blocking PnL. Takes 8 seconds to give pnl and acc summary
from ib_insync import IB
import json

market = 'nse'

# ...variables initialization
with open('var.json', 'r') as fp:
    data = json.load(fp)

host = data['common']['host']
port = data[market]['port']
cid = 0

# ...connect to IB
ib = IB().connect(host=host, port=port, clientId=cid)
acct = ib.managedAccounts()[0]

# ..get account summary
accsum = ib.accountSummary(account=acct)

# ..get liquidity and funds dictionary
funds = {t.tag: t.value
         for t in accsum
         if t.tag in ["NetLiquidation", "AvailableFunds"]}

# ..dailyPnl
ib.reqPnL(acct)
ib.sleep(8)
pnlobj = ib.pnl()[0]
ib.cancelPnL(acct)

pnldict = {'dailyPnL': pnlobj.dailyPnL,
           'unrealizedPnL': pnlobj.unrealizedPnL,
           'realizedPnL': pnlobj.realizedPnL}

print({**funds, **pnldict})
