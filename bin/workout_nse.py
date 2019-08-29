# workout_nse.py

"""Program to prepare closing (BUY) trades
Date: 23-July-2019
Ver: 1.0
Time taken: 1 min
"""

from z_helper import *

# from json
a = assign_var('nse') + assign_var('common')
for v in a:
    exec(v)
    
def workout_nse(ib):
    '''Program to dynamically prepare closing trades
    Args:
        (ib) as connection object
    Returns:
        tuple of buy contracts and orders'''
    
# with get_connected('nse', 'live') as ib:
    pos = ib.positions()
    trades = ib.trades()
    pos_contracts = ib.qualifyContracts(*[Contract(conId=c) for c in [p.contract.conId for p in pos]])
    pos_tickers = ib.reqTickers(*pos_contracts)
    pos_prices = {t.contract.conId: t.marketPrice() for t in pos_tickers}    
    
    #... position dataframe
    pos_cols = ['conId', 'symbol', 'localSymbol', 'secType', 'lastTradeDateOrContractMonth', 'strike', 'right']
    pos_df = util.df(p.contract for p in pos)[pos_cols]
    pos_df = pos_df.rename({'lastTradeDateOrContractMonth': 'expiry'}, axis='columns')
    pos_df = pos_df.assign(dte=pos_df.expiry.apply(get_dte))
    pos_df['position'] = [p.position for p in pos]
    pos_df['avgCost'] = [p.avgCost for p in pos]
    pos_df['close'] = pos_df.conId.map(pos_prices)

    # take maximum of precision and minimum of hvstPrice, avgCost
    pos_df = pos_df.assign(hvstPrice=np.maximum(np.minimum((pos_df.dte.apply(hvstPricePct)*pos_df.avgCost).apply(lambda x: get_prec(x, 0.05)), pos_df.close), prec))

    if trades:
        trades_df = util.df(t.contract for t in trades).join(util.df(t.order for t in trades)).join(util.df(t.orderStatus for t in trades), lsuffix='_')
        trades_cols = ['conId', 'symbol', 'localSymbol', 'secType', 'expiry', 'strike', 'right', 'undPrice', 'sd', 
                       'rom', 'orderId', 'permId', 'action', 'totalQuantity', 'close', 'lmtPrice', 'status']
    else:
        trades_df = pos_df.assign(action = 'No action', status = 'No status')
        trades_cols = ['conId', 'symbol', 'localSymbol', 'secType', 'expiry', 'strike', 'right', 'undPrice', 'sd', 
                       'rom', 'action', 'status']    

    # join with other parameters in sized_nse.pkl
    trades_df = trades_df.set_index('conId').join(pd.read_pickle(fspath+'sized_nse.pkl')[['optId', 'close', 'undPrice', 'rom', 'stDev']].set_index('optId'), lsuffix='_').rename_axis('conId').reset_index()
    trades_df = trades_df.rename({'lastTradeDateOrContractMonth': 'expiry'}, axis=1)
    trades_df = trades_df.assign(sd=abs(trades_df.strike-trades_df.undPrice)/trades_df.stDev)

    trades_df = trades_df[trades_cols]
    
    #... check for margin breach
    ac_df=util.df(ib.accountSummary())[['tag', 'value']]

    if (float(ac_df[ac_df.tag == 'AvailableFunds'].value.values) /
        float(ac_df[ac_df.tag == 'NetLiquidation'].value.values)) > (1-ovallmarginlmt):
        marginBreached = True
    else:
        marginBreached = False
        
    if marginBreached:
        cancel_sells(ib)

    #... place BUY orders for positions without BUY orders

    # get the pos conIds that have some trade (BUY or SELL)
    pos_trade_cids = [tid for tid in trades_df.conId if tid in list(pos_df.conId)]

    # remove pos conIds that have BUY trade action and which are active
    pos_buy_cids = list(trades_df[trades_df.conId.isin(pos_trade_cids) & 
                                  (trades_df.action == 'BUY') & 
                                  ~(trades_df.status.isin(['Cancelled', 'PendingCancel' ]))].conId)
    pos_buy_df = pos_df[~pos_df.conId.isin(pos_buy_cids)]

    # make lot and qty for trade blocks and rename conId to optId
    pos_buy_df = pos_buy_df.assign(lot=pos_buy_df.position.apply(abs), qty=1, expPrice=pos_buy_df.hvstPrice)
    pos_buy_df.rename(columns={'conId': 'optId'}, inplace=True)

    # make the buy back expPrice slightly lesser for those whose close price == buyback expPrice
    mask = (pos_buy_df.expPrice == pos_buy_df.close) & (pos_buy_df.expPrice > 0.05)
    pos_buy_df.loc[mask, 'expPrice'] = pos_buy_df[mask].expPrice-0.05

    # Make the NaN's to 0.05 expPrice
    pos_buy_df.loc[pos_buy_df.expPrice.isnull(), 'expPrice'] = 0.05

    # make the expPrice to the correct precision - else trades will fail
    pos_buy_df = pos_buy_df.assign(expPrice=pos_buy_df.expPrice.apply(lambda x: get_prec(x, prec)))
    
    return pos_buy_df

#_____________________________________

