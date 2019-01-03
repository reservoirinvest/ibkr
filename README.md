# Objectives

* Have a framework for analysis and automation of trades in Interactive Brokers

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
* Harvests open option positions from a linest curve
* Prepares to Sow
   * Checks available funds
   * Makes a *blacklist* (existing positions which have run over position limit)
   * Focuses on Puts
      * with Strikes above the mean
   * Filters based on expected rom
   * Checks consumption of funds
 * Places **Harvests** (closing trades) and **Sows** (Opening Trades)
 * Records the harvests and sows
   