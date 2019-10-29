from ib_insync import *

def get_acc_summary(ib):
	async def pnlcoro(ib):
		'''Gets the pnl object'''
		acct = ib.managedAccounts()[0]
		pnl = ib.reqPnL(acct)
		await ib.pnlEvent
		return pnl
	
	# get the account summary
	df_ac = util.df(ib.accountSummary())
	NLV = float(df_ac[df_ac.tag.isin(['NetLiquidation'])].value.iloc[0])
	initMargin = float(df_ac[df_ac.tag.isin(['InitMarginReq'])].value.iloc[0])
	unrealPnL = float(df_ac[df_ac.tag.isin(['UnrealizedPnL'])].value.iloc[0])
	realPnL = float(df_ac[df_ac.tag.isin(['RealizedPnL'])].value.iloc[0])
	avFunds = float(df_ac[df_ac.tag.isin(['AvailableFunds'])].value.iloc[0])
	acsum = {"NLV": NLV, "initmargin": initMargin, "unrealzPnL": unrealPnL, 
	 "realzPnL": realPnL, "avFunds": avFunds}

	pnl = ib.run(pnlcoro(ib))

	pnldict = pnl.dict()
	del pnldict['modelCode']

	pnldict.update(acsum)

	return pnldict

with IB().connect('127.0.0.1', 1300, 0) as ib:
	# pnldict = get_acc_summary(ib)
	acct = ib.managedAccounts()[0]
	pnl = ib.reqPnL(acct)
	print(pnl)
	
# # print(pnldict)

# import PyQt5.QtWidgets as qt

# class AcSummTbl(qt.QTableWidget):

# 	headers = ['Item': 'Value']
	
# 	def __init__(self, parent=None):
# 		qt.QTableWidget.__init__(self, parent)
		
	
# 	for i, j in enumerate(
	

