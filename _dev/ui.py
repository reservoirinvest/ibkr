import json
from quart import Quart
from ib_insync import IB, util

host = 3000 # <3000 for nse> | <1300 for snp>
cid = 0

""" app = Quart(__name__)

@app.route('/pnl')
async def pnl():
    with await IB().connectAsync('127.0.0.1', host, cid) as ib:
        acct = ib.managedAccounts()[0]
        pnl = ib.reqPnL(acct)
        await ib.pnlEvent
    resp = json.dumps(util.tree(pnl))
    return resp
 """
# app.run()

ib = IB().connect('127.0.0.1', 3000, 0)

async def pnlcoro(ib):
    '''Gets the pnl object'''
    acct = ib.managedAccounts()[0]
    pnl = ib.reqPnL(acct)

    return pnl

async def accsum(ib):
    # get the account summary

    # pnl = ib.run(pnlcoro(ib))
    pnl = await pnlcoro(ib)

    df_ac = util.df(ib.accountSummary())

    NLV = float(df_ac[df_ac.tag.isin(['NetLiquidation'])].value.iloc[0])
    initMargin = float(df_ac[df_ac.tag.isin(['InitMarginReq'])].value.iloc[0])
    unrealPnL = float(df_ac[df_ac.tag.isin(['UnrealizedPnL'])].value.iloc[0])
    realPnL = float(df_ac[df_ac.tag.isin(['RealizedPnL'])].value.iloc[0])
    avFunds = float(df_ac[df_ac.tag.isin(['AvailableFunds'])].value.iloc[0])
    acsum = {"NLV": NLV, "initmargin": initMargin, "unrealzPnL": unrealPnL, 
        "realzPnL": realPnL, "avFunds": avFunds}

    pnldict = pnl.dict()
    del pnldict['modelCode']

    acsum.update(pnldict)

    return accsum

acsum = ib.run(accsum(ib))
	
print(acsum)