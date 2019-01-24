# Objective

* Have a framework for analysis and automation of trades in Interactive Brokers

# Goals
1) The framework should be able to:
* Scan - build the target scrips, extract underlyings, extract options and pickle
* Manage 
    * manual - decide the _strategy_, close (harvest) trades, open (sow) new trades
    * auto - high frequency automatic trading with error management and exception management
* Analyze - visual tool to quickly assemble data for analysis and decision making

2) Use best programming best practice
    * use native IBKR information, with no/little dependancies from external sites/sources

3) Have technical capability to do everything in a cloud infrastructure, with monitoring on smart phone

# Strategy
## Closing
1) Dyamically determine the closing price percentage for short options based on dte
2) Determine harvest price from minimum of determined closing price and marketprice

## Opening
### Puts


# Markets
* nse - for India
* snp - for US

# NSE

## 01_nse_scan

* Extract symbols and margins from (5paisa)[https://www.5paisa.com/5pit/spma.asp]
* Translates to IBKR symbols
* Gets put price, call price and delta from Black-Scholes 
* Prepares to *pickle* 
   * Underlying symbols dictionary with:
      * volatility, hi52, lo52, average, dividend
      * integrated with lots and margins
      * pickles them to *\_symbol.pkl* file
   * Option chains dataframe with:
      * option chain tickers (with expiries and strikes that fall in 2 std deviation from underlying)
      * option greeks (black-scholes delta for probability-of-profit *pop*)
      * expected price (adjusted for premiums / penalties and base decimals)
      * margins (for return-on-margin *rom*)
      * pickles the dataframe to *symbol.pkl* file
* Adjusts days-to-expire for last day option expiries
   
## 02_nse_manage

* Reads the account summary
* _Harvests_ open option positions from a linest curve
* Prepares to _Sow_
   * Checks available funds
   * Makes a *blacklist* (existing positions which have run over position limit)
   * Focuses on Puts
      * with Strikes above the mean
   * Filters based on expected rom
   * Checks consumption of funds
 * Places **Harvests** (closing trades) and **Sows** (Opening Trades)
 * Records the harvests and sows
   