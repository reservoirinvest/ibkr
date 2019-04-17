# 1Scrip module

## Functions

### get_margins(ib, contracts, * lotsize)
* Not for nse equities due to subscription constraints

### get_snps(ib) or get_nses(ib)
* generates a list of underlying contracts

### get_dte(dt)

### get_rollingmax_std(ib, c, dte, durmult=3)
* standard deviation to limit option contract generation

### get_maxfallrise(ib, c, dte)
* gets the maximum fall and rise with 52-week high and low

### catch(func, handle=lambda e : e, * args, ** kwargs)
* allows list comprehesion to ignore errors

### save_open_orders(ib, fspath='.../data/snp/')
* to save any open orders that are mass cancelled to enable margin checks

### get_portfolio(ib)
* produces datafram with margins and assignments
**DEPENDENCIES:** get_margins()

### get_p_remqty(ib)
* produces remaining quantities from existing positions based on margins.
* zeros are to be blacklisted
**DEPENDENCIES:** save_open_orders(), get_portfolio()

### get_snp_options(ib, undContract, undPrice)  or get_nse_options(ib, undContract, undPrice)
* produces a list of option in pickles
**DEPENDENCIES:** get_maxfallrise(), get_rollingmax_std(), get_margins()

### get_connected(market, trade_type)
* connects to appropriate ib client

### get_snp_remqty(ib, remqty_p, undContracts) or get_nse_remqty(ib, remqty_p, undContracts)
* gets the remaining quantities of each underlying contract after checking with portfolio

## Scripts


