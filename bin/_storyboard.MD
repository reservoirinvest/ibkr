# NOTEBOOKS (.ipynb)
* <b>helper.ipynb</b>    --- for generic helper functions

* <b>nse_func.ipynb</b>  --- for nse specific functions
* <b>nse_main.ipynb</b>  --- for nse management and control

* <b>snp_func.ipynb</b>  --- for snp specific functions
* <b>snp_main.ipynb</b>  --- for snp management and control

* <b>_jup2py.ipynb</b>   --- for auto-converting jupyter to py modules

* <b>ztest.ipynb</b>     --- for testing functions
  * test notebook should not have first line of a cell ending with .py extension
  
# PYTHON (.py)
* <b>helper.py</b>   --- auto-generated from helper.ipynb
Contains the common helper functions

* <b>nse.py</b>      --- auto-generated from nse_func.ipynb
* <b>snp.py</b>      --- auto-generated from snp_func.ipynb
Contains nse, snp specific helper functions

* <b>nse_main.py</b>
* <b>snp_main.py</b>
The main programs

  
# FIELD-LIST
## nses.pickle
['contract', 'lot', 'margin', 'undPrice']

## (symbol).pkl
['symbol', 'expiration','strike', 'dte', 'undPrice', 'lo52', 'hi52', 'Fall', 'Rise', 'loFall', 'hiRise', 
'std3', 'loStd3', 'hiStd3', 'optId', 'right', 'optPrice', 'lotsize', 'optMargin', 'rom']

## target.pickle
['optId', 'symbol', 'right', 'expiration', 'dte', 'strike', 'undPrice',  'lo52', 'hi52', 'Fall', 'Rise', 'loFall', 'hiRise', 
'std3', 'loStd3', 'hiStd3', 'optId', 'right', 'optPrice', 'lotsize', 'optMargin', 'rom', 'expPrice', 'remQty', 'expQty']


# NSE - PROCESS

## Every weekend
* Delete all files in data\nse
* Generate nses and pickle it
* Generate options and pickle them

### Programs
p_nses
p_nseopts

## Every day
* Upload nses.pickle
* Update undMargin and undPrices in it
* Save back the nses.pickle

* Upload the options pickles
* Update the undPrice, optPrice, optMargin, rom

* Generate the <b>target</b> options
   * Check <b>remQty</b>
   * Start with 1 option contract per strike+expiry
   * Determine the appropriate <b>expPrice</b>
   
### Programs
upd_nses
upd_opt (from helper)
remqty_nse
   
## Every minute
* Update optPrice for the <b>target</b> options
* If a trade fills:
   * re-calculate <b>remQty</b>
   * remove blacklists if any in the <b>target</b> option list
   * increment the <b>expPrice</b> to a higher threshold to reduce risk / get better reward
   
************************************************************************************

# NSE Functions

### p_nses              *pickle underlyings*
(ib) ---> <nses>.pkl 

### p_nseopts        *pickle nse options* 
(ib, undContract, undPrice, lotsize, margin) ---> <symbol>.pkl

### upd_nses          *update nses - undPrice and margin*
(ib, nses) ---> <nses>.pkl                                                   

### remqty_nse       *determine the remaining quantities*
(ib, df_nses) ---> dictionary {symbol: remQty}

### target_nse        *make a target of options with expected price*
(ib)  --->  df
* Filters the top 5 of each symbol whose strikes are beyond 52 week hi (for calls) and 52 week lo (for puts)
* Removes blacklisted symbols through remaining quantity (remqty)
* Generates expPrice based on minimum RoM set
* For expPrice < 0.15 (minOptPrice) sets it to that
* Computes the expected RoM from the expPrice
* Pickles the target

### manage_nse      *manages nse by the minute* (INCOMPLETE)
* gets the target dataframe from target_nse
* places the orders
* pickles the trades in *_tradesDone.pickle*

Should ideally:
 - cancel the openOrders
 - place the harvestOrders
 - place the sowOrders

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

************************************************************************************

# TO-DO LIST

## Helper Functions

### blks
(ib, func)  ---> None
* Performs the function in blocks to prevent Start/Stop threshold errors

### logger
* To differentiate between ib_insync's internal logs and INFO generated on purpose 
**************************************************************************************



