# target_nse.py

"""Program to create targets form sized pickles
Date: 23-July-2019
Ver: 1.0
Time taken: milliseconds
"""

from z_helper import *

# from json
a = assign_var('common') + assign_var('nse')
for v in a:
    exec(v)

def target_nse(ib, df_sized, blacklist):
    '''Args:
        (ib) as connection object
        (df_sized) picked up from sized pickle
        (blacklist) as list of blacklisted stocks
       Returns:
        tb: trade blocks and pickles into targets.pickle
    '''
    
    # from portfolio
    #_______________
#     with get_connected('nse', 'live') as ib:
    p = util.df(ib.portfolio()) # portfolio table

    # extract option contract info from portfolio table
    dfp = pd.concat([p, util.df([c for c in p.contract])[util.df([c for c in p.contract]).columns[:7]]], axis=1).iloc[:, 1:]
    dfp = dfp.rename(columns={'lastTradeDateOrContractMonth': 'expiration'})

    # get the total position
    dfp1 = dfp.groupby('symbol').sum()['position']

    # from options pickle
    #____________________

    # get the options
    df_opt = df_sized.assign(und_remq=(nse_assignment_limit/(df_sized.lot*df_sized.undPrice)).astype('int')) # remaining quantities in entire nse

    # remove nan in margins and close prices. These could be dead ones.
    df_opt = df_opt[~df_opt.margin.isnull()].reset_index(drop=True)
    df_opt = df_opt[~df_opt.close.isnull()].reset_index(drop=True)

    # remove margins with 1e7
    df_opt = df_opt[df_opt.margin < 1e7]

    # establish quantity and minimum expected price and sdmult
    df_opt = df_opt.assign(qty=1, expPrice = df_opt.close+0.1, sd=abs(df_opt.strike-df_opt.undPrice)/df_opt.stDev)

    # recacluate rom based on expPrice upgrade
    df_opt = df_opt.assign(rom=df_opt.expPrice/df_opt.margin*365/df_opt.dte*df_opt.lot)

    # sort the standard deviation change (lowest sd is most risky)
    df_opt = df_opt.sort_values('sd', ascending=True)

    # for those not meeting minimum expected ROM, up the expected price
    rom_mask = (df_opt.rom < minexpRom)
    df_opt.loc[rom_mask, 'expPrice'] = ((df_opt[rom_mask].expPrice * minexpRom )/ df_opt[rom_mask].rom).apply(lambda x: get_prec(x, prec))

    # filter based on remaining quantity
    #___________________________________

    # compute the remaining quantities
    df_opt1 = df_opt.groupby('symbol').first()[['lot', 'margin', 'und_remq']]

    df_opt2 = df_opt1.join(dfp1).fillna(0).astype('int')
    df_opt2 = df_opt2.assign(remqty=df_opt2.und_remq+(df_opt2.position/df_opt2.lot).astype('int'))

    dfrq = df_opt2[['remqty']]

    # remove existing positions with remqty < 1
    blacklist = blacklist + list(dfrq[dfrq.remqty < 1].index)
    df_opt = df_opt[~df_opt.symbol.isin(blacklist)]

    # pd.options.display.float_format = '{:,.2f}'.format
    cols = ['symbol', 'undId', 'optId', 'dte', 'lot', 'right', 'undPrice', 'strike', 'stDev', 'lo52', 'hi52', 'margin', 'qty', 'close', 'rom', 'sd', 'expPrice']
    df_opt[cols].to_pickle(fspath+'targets.pkl')
    
    return df_opt[cols]

#_____________________________________

