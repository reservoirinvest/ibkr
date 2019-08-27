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

def target_nse(ib, df_sized, blacklist, assignment_limit):
    '''Args:
        (ib) as connection object
        (df_sized) picked up from sized pickle
        (blacklist) as list of blacklisted stocks
        (assignment_limit) nse_assignment_limit from variables.json
       Returns:
        tb: trade blocks and pickles into targets.pickle
    '''
    
    # get remaining quantities
    dfrq1 = dfrq(ib, df_sized, assignment_limit, exchange)

    # get the options
    df_opt = df_sized.assign(und_remq=assignment_limit/(df_sized.lot*df_sized.undPrice)) # remaining quantities in entire nse

    # integrate with remq
    df_opt = df_opt.set_index('symbol').join(dfrq1.remq)

    # for those without remq, replace with und_remq
    df_opt.loc[df_opt.remq.isnull(), 'remq'] = df_opt[df_opt.remq.isnull()].und_remq

    # weed out options with remq lesser than 1
    df_opt = df_opt[df_opt.remq >= 1]

    # make remaining quantities an int
    df_opt = df_opt.assign(remq=df_opt.remq.astype('int32'), und_remq=df_opt.und_remq.astype('int32'))

    # remove nan in margins and close prices. These could be dead ones.
    df_opt = df_opt[~df_opt.margin.isnull()]
    df_opt = df_opt[~df_opt.close.isnull()].reset_index()

    # remove margins with 1e7
    df_opt = df_opt[df_opt.margin < 1e7]

    # remove blacklisted options
    df_opt = df_opt[~df_opt.index.isin(blacklist)]

    # establish quantity and minimum expected price and sdmult
    df_opt = df_opt.assign(qty=1, expPrice = df_opt.close+0.1, sd=abs(df_opt.strike-df_opt.undPrice)/df_opt.stDev)

    # recacluate rom based on expPrice upgrade
    df_opt = df_opt.assign(rom=df_opt.expPrice/df_opt.margin*365/df_opt.dte*df_opt.lot)

    # sort the standard deviation change (lowest sd is most risky)
    df_opt = df_opt.sort_values('sd', ascending=True)

    # for those not meeting minimum expected ROM, up the expected price
    rom_mask = (df_opt.rom < minexpRom)
    df_opt.loc[rom_mask, 'expPrice'] = ((df_opt[rom_mask].expPrice * minexpRom )/ df_opt[rom_mask].rom).apply(lambda x: get_prec(x, prec))

    # remove targets with expPrice 0.0. This is caused by negative margins
    df_opt.loc[df_opt.expPrice < minexpOptPrice, 'expPrice'] = minexpOptPrice

    # symbols busting remaining quantity limit
    d = {'qty': 'sumOrdQty', 'remq': 'remq'}
    df_bustingrq = df_opt.groupby('symbol').agg({'qty': 'sum', 'remq': 'mean'}).rename(columns=d)
    df_bustingrq = df_bustingrq[df_bustingrq.sumOrdQty > df_bustingrq.remq].reset_index()

    # pd.options.display.float_format = '{:,.2f}'.format
    cols = ['symbol', 'undId', 'optId', 'dte', 'lot', 'right', 'undPrice', 'strike', 'stDev', 'lo52', 'hi52', 'margin', 
            'qty', 'remq', 'close', 'rom', 'sd', 'expPrice']
    
    df_opt[cols].to_pickle(fspath+'targets.pkl')
    return df_opt[cols]

#_____________________________________

