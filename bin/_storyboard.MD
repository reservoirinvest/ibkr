# FLOW

## Build (before trading day)
Get the **Base** underlyings ---> Get the **Option** prices---> Get the **Remaining Quantities** --->
---> Establish the **targets** with **expected price** and **expected rom**

If there are no sufficient targets, adjust JSON **variables** and re-run the above

Cancel **ALL Orders** ---> Place the **Closing Orders** ---> Place the **Opening Orders** for **targets**

## Manage (*live* trading)

1. If *InitMarginReq* is 85% of *NetLiquidation* **cancel ALL Opening orders** 

2. Upon change in portfolio item (or) order fill event:

   Check *InitMarginReq* against *NetLiquidation*. 
   If it is ok:
      For the **symbol filled** --->**Cancel openorders** ---> Place **Closing orders** --> Recalculate **price** and **margins** ---> 
      Recalculate **Remaining Quantities** ---> Update the **targets** ---> Place **Opening Orders** if not blacklisted
      
   Else **cancel ALL Opening orders**

***********************************************************

# NOTEBOOKS (.ipynb)

## Common functions
* <b>helper.ipynb</b>    --- for generic helper functions
* <b>_jup2py.ipynb</b>   --- for auto-converting jupyter to py modules

## Market specific functions

* <b>nse_func.ipynb</b>  --- for nse specific functions
* <b>nse_main.ipynb</b>  --- for nse management and control

* <b>snp_func.ipynb</b>  --- for snp specific functions
* <b>snp_main.ipynb</b>  --- for snp management and control

...to ensure that the functions are picked up, the first line in the cell should be:
 #<function_name>.py

## Experiments & Junk files

* <b>__scratchpad.ipynb</b> --- for scratchpad
* <b>ztest.ipynb</b>     --- for testing functions

****************************************************
  
# PYTHON (.py)
* <b>helper.py</b>   --- auto-generated from helper.ipynb
   ...contains the common helper functions

* <b>nse.py</b>      --- auto-generated from nse_func.ipynb
* <b>snp.py</b>      --- auto-generated from snp_func.ipynb
   ...contains nse, snp specific helper functions

* <b>nse_main.py</b>
* <b>snp_main.py</b>
  ...contains main programs

*****************************************************

# JSON
* <b>variables.json<b> containing variable assignments for the programs
  
# FIELD-LIST

## base.pickle

Contains qualified underlyings with strikes, expiries and lots
['symbol', 'expiration', 'strike', 'right', 'lot', 'und_remq', 'dte', 'cid', 
'undPrice', 'lo52', 'hi52', 'Fall', 'Rise', 'Std',  'loStd', 'hiStd']

## opts.pickle

Contains computed margin and rom for the option
['symbol', 'expiration', 'strike', 'right', 'lot', 'dte', 'cid', 'undPrice', 'lo52', 'hi52', 
'Fall', 'Rise', 'Std', 'loStd', 'hiStd', 'optId', 'optPrice', 'margin', 'rom']

## targets.pickle

Same as opts, but cleaned up for margin and rom with remqty, expPrice and expRom
['symbol', 'expiration', 'strike', 'right', 'lot', 'dte', 'cid', 'undPrice', 'lo52', 'hi52', 
'Fall', 'Rise', 'Std', 'loStd', 'hiStd', 'optId', 'optPrice', 'margin', 'rom', 'remqty', 'expPrice', 'expRom']
   
************************************************************************************

# NSE Functions

### imports.py
Has all the imports needed
Assigns variables from JSON

### base.py                 *creates the base.pickle*
(ib) ---> base.pickle

### opts.py                 *computes margin and rom* 
(ib) ---> opts.pickle

### remqty_nse.py     *determine the remaining quantities*
(ib) ---> dataframe with symbol: remqty. Indexed on symbol

### targets.py              *generates the targets*
(ib) ---> targets.pickle                                                  
* Filters the top 5 of each symbol whose strikes are beyond 52 week hi (for calls) and 52 week lo (for puts)
* Removes blacklisted symbols through remaining quantity (remqty)
* Generates expPrice based on minimum RoM set
* For expPrice < 0.15 (minOptPrice) sets it to that
* Computes the expected RoM from the expPrice
* Pickles the target

### gen_expPrice.py    *generates expected price & expected rom*
(ib) ---> repickles target.pickle

### upd.py                   *update nses - undPrice and margin*
(ib) ---> targets.pickle

### watchlists.py         *generate watch-list and target-list in csv*
(ib)  ---> watch.csv targets.csv

### dynamic.py            *places trades with reduced risk*

***********************************************************************************

# Helper Functions

### get_connected
(market, trade_type) ---> ib as a connection object
* connects to appropriate ib client

### get_dte
(dt) ---> days as integer
* gets number of days to expiry from date given

### get_rollingmax_std
(ib, c, dte, durmult=3) ---> df_stdev DataFrame
* standard deviation to limit option contract generation

### get_maxfallrise
(ib, c, dte) ---> (lo52, hi52, max_fall, max_rise) as a tuple
* gets the maximum fall and rise with 52-week high and low

### catch
(func, handle=lambda e : e, * args, ** kwargs) ---> successful function or np.nan for exceptions
* allows list comprehesion to ignore errors

### save_open_orders
(ib, fspath='.../data/snp/') ---> dfopen as a DataFrame of openorders. Also pickles to *_openorders.pkl*
* saves any open orders that are mass cancelled to enable margin checks
* this could be used to 'write-back' any openorders that were manually generated

### upd_opt      *update options - price*
(ib, symbol) ---> <symbol>.pkl
    
### grp_opts      *group options based on sorts, strikes, puts and calls*
(df)   ---> df

### get_prec       *get precision value w.r.t. a base entered*
(v, base) ---> returns the precise value

### assign_var      *declarations to assign variables for nse / snp*
(market) ---> returns a variable list (varList).
These need to be included on top of modules with the following code:
   a = assign_var('nse')
   for v in a:
         exec(v)
         
### hvstPricePct   *getting harvest price percentage, based on curve*
(dte)  ---> returns the price percentage to harvest

### trade_blocks   * makes blocks for placing trades*
(ib, df)    ---> returns blocks for placing trades

### doTrades         *place the trades*
(ib, coblks)  ---> returns list of trades done

************************************************************************************