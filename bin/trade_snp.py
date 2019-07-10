# trade_snp.py
from z_helper import *

# from json
a = assign_var('snp') + assign_var('common')
for v in a:
    exec(v)

# extract from the pickle
df_opt = pd.read_pickle(fspath+'sized.pkl')

# remove nan in margins and close prices. These could be dead ones.
df_opt = df_opt[~df_opt.margin.isnull()].reset_index(drop=True)
df_opt = df_opt[~df_opt.close.isnull()].reset_index(drop=True)

# remove margins with 1e7
df_opt = df_opt[df_opt.margin < 1e7]

# establish quantity and minimum expected price and sdmult
df_opt = df_opt.assign(qty=1, expPrice = df_opt.close+0.1, sd=abs(df_opt.strike-df_opt.undPrice)/df_opt.stDev)

# recacluate rom based on expPrice upgrade
df_opt = df_opt.assign(rom=df_opt.expPrice/df_opt.margin*365/df_opt.dte*df_opt.lot)

# calculate the standard deviation change (lowest is most risky)
df_opt = df_opt.sort_values('sd', ascending=True)

# for those not meeting minimum expected ROM, up the expected price
rom_mask = (df_opt.rom < minexpRom)
df_opt.loc[rom_mask, 'expPrice'] = ((df_opt[rom_mask].expPrice * minexpRom )/ df_opt[rom_mask].rom).apply(lambda x: get_prec(x, prec))

#_____________________________________

