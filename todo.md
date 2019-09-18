# The To-do list

## Today
* async code for margin and price for all option contracts

## Filtering
* From OHLCs - use Bollinger Bands + Keltner Channel + TRIX + Ultimate Oscillator

## Overall
* Masks for filters. Test it with ohlcs.pkl

* Function to cancel uncovered SELLs
* Function to write covered PUTs for shorts and CALLs for longs in SNP
* Function to adjust n-band to remaining quantities
* Function for last day trades
* Function for NLV, Initial Margin and Composite ROI computation
* Function for naked mean and median RoM and PoP from target
* Function for Event triggered 'dynamic' workout
* Function to historize scrips for specific Put and Call SD for 60 continuous days for both SNP and NSE in data>history
* 'Cryptic' dashboard for positions (covered and uncovered), open trades, P&L and overall-limits
* Optimize pickles by removing computed fields
* Make a json list of scrips with physical delivery for NSE
* Make last day trade function - with no BUYs for the last day trade and BUYs only for ITM with physical delivery in NSE
* Function to adjust ROI and POP based on expiry date
* Combine sized, target and workout to one program for both SNP and NSE
## Strategy
* Make different strategy functions and test
## GUI
* Learn about plotting with MatPlotLib - use datascience
* Build a GUI (EEL with SVG and JS or pyQt5 - Ewald)