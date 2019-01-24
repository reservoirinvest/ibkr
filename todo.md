# The To-do list

* Clean-up scan pickle outputs
    * Make it purely data extracts:
        * Remove ROM and other calculations
    * Have separate df for ohlc and underlying data - to optimize space
    * Make _assembler functions_ to get required data from the pickles
* Remake manage using the assembler functions
* Make error detection and continuity robust
    * learn how to use callbacks / error events
    * use timer for very long no-data issues (e.g. pickle outputs)

# Levels of data
## OHLC
* Filename: ``<symbol>_`` ohlc.pkl
* Fields: xdate(index), symbol, o, h, l, c
   
## Underlying
* Filename: `_`underlying.pkl
* Fields: xtime, symbol, strikes, expiries, divrate, undPrice, margin

## Options
Stores the strikes and expiries of qualified options
* Filename: ``<symbol>_``opt.pkl
* Fields: symbol, strike, expiry, type, price, rom, pop

# Helper functions
* *get_stdev(days=365, value=c)* - to get 365 day standard deviation from close values of ohlc 
* *get_bsm(undPrice, strike, dte, rate, volatality, divrate)* - to get blackcholes put price, call price and delta
* *catch(func)* - to catch errors in list comprehension
* *get_dividend_ticker(contract)* - to get the dividend of the contract object with conId
* *expPricePct(expiry)* - to get expected price percentage based on expiry for an option based on linest