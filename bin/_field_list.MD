# Output
* symbol, strike, expiry, right, expPrice, expROM, POP, remQty, optContract

# Intermediates

## Filter
* optChain, undPrice, strike, stdev, dte = Filtered list

## ROM
* undPrice, dte, margin = ROM

## POP
* (from program)
    * get_rfrate
    * get_dividends
    * undPrice, strike, dte, rfrate, volatility, dividend = bsmPrice, bsmDelta (POP)

# Inputs

## Generic
* (from helper programs): 
   * get_connected
   * get_margins
      * existing option margins

   * get_underlyings (qualified)
   
   * lotsize
   
   * get_dte()
   
   * remQty
   * blacklist
   
   
* (from csv):
   * putCutoff, callCutoff, RiskPenalty
   
* (assembler)
    * undMargin
    

## Underlying
* (from program)
    * ohlc
    * undPrice, option_chain (s+e), option_margin, 

## Options
* symbol(filtered), strike, expiry, dte, stdev, volatility, optPrice, optMargin, optBid, optAsk, = (optPrice, optMargin)

# Helper functions
* Delete only old pickle files
* Convert jupyter json cells to py programs

# Learn
* Using asyncio on function with wait
* Using asyncio on function with gather
* Using asyncio in for loop

