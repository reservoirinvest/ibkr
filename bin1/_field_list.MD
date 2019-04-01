# Output
* symbol, strike, expiry, right, expPrice, expROM, POP, remQty, optContract

# Intermediates

## Filter
* optChain, undPrice, strike, stdev, dte = Filtered list

## ROM
* undPrice, dte, margin = ROM

## POP
* undPrice, strike, dte, rate, volatility, dividend = bsmPrice, bsmDelta (POP)

# Inputs

## Generic
* (from program): 
   * get_connected
   * get_margins
      * existing option margins
   
   * get_rfrate
   * get_underlyings (qualified) 
   
   * undMargin
   
   * lotsize
   
   * get_remQty
   * get_blacklist
   
* (from csv):  
   * putCutoff, callCutoff, RiskPenalty

## Underlying
* (from program)
    * ohlc
    * undPrice, option_chain (s+e), dividends, option_margin, 

## Options
* symbol(filtered), strike, expiry, dte, stdev, volatility, optPrice, optMargin, optBid, optAsk, = (optPrice, optMargin)

# Helper functions
* Delete only old pickle files
* Convert jupyter json cells to py programs

mastermargin

