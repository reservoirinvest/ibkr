import pandas as pd
import numpy as np

#***      Function to print python code in Jupyter   ****
#________________________________________________________

def display_py(code):
    """Displays python file code in Jupyter

    Arg: (string from py file) code

    Output: code formatted for jupyter

    Usage:
    with open(myfile) as f:
         code = f.read()

    display_py(code)
    """
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter
    import IPython
    
    formatter = HtmlFormatter()

    html_code = highlight(code, PythonLexer(), HtmlFormatter())
    styled_html = '<style type="text/css">{}</style>{}'.format(formatter.get_style_defs('.highlight'), html_code)
    ipython_code = IPython.display.HTML(styled_html)

    return ipython_code

#***   Error catching for list comprehension ***
#_______________________________________________

def catch(func, handle=lambda e : e, *args, **kwargs):
    '''List comprehension error catcher
    Args: 
        (func) as the function
         (handle) as the lambda of function
         <*args | *kwargs> as arguments to the functions
    Outputs:
        output of the function | <np.nan> on error
    Usage:
        eggs = [1,3,0,3,2]
        [catch(lambda: 1/egg) for egg in eggs]'''
    try:
        return func(*args, **kwargs)
    except Exception as e:
        np.nan

#***      Function to get historical data     *****
#___________________________________________________
def get_hist(ib, contract, duration):
    '''Gets 1-day bars of contracts for the duration specified
    Args:
        (contract): contract as obj
        (duration): history days as int
    Returns: dataframe of symbol, date, ohlc, avg and volume 
    '''

    # Prepare the duration
    strduration = str(duration) + ' D'

    # Extract the history
    hist = ib.reqHistoricalData(contract=contract, endDateTime='', 
                                    durationStr=strduration, barSizeSetting='1 day',
                                                whatToShow='Trades', useRTH=True)

    # Make the dataframe
    cols=['ibSymbol', 'D', 'O', 'H', 'L', 'C', 'Avg', 'Vol']
    df = pd.DataFrame([(contract.symbol, h.date, h.open, h.high, h.low, 
                       h.close, h.average, h.volume) 
                      for h in hist], columns=cols)
    return df

#***         Function to get volatilities     *****
#__________________________________________________

def maxvol(ib, contract):
    '''Gets maximum volatailty
       Args:
          (ib) as object for keeping ibkr connection
          (contract) as object
       Returns: maxvol as float - maximum of 12 month's HV and IV '''
    bars1 = ib.reqHistoricalData(contract, endDateTime='', durationStr='12 M',
            barSizeSetting='1 day', whatToShow='OPTION_IMPLIED_VOLATILITY', useRTH=True)
    bars2 = ib.reqHistoricalData(contract, endDateTime='', durationStr='12 M',
            barSizeSetting='1 day', whatToShow='HISTORICAL_VOLATILITY', useRTH=True)
    maxvol = max([b.close for b in bars1]+[b.close for b in bars2])
    return maxvol

#***  Get the underlying's price
def getprice(ib, contract):
    '''Gets the price of the contract
    Arg:
       (ib) as object for keeping ibkr connection
       (contract) as object
    Returns: close priceas float'''
    
    bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='60 S',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1,
            keepUpToDate=True)[-1:]
    
    close = [catch(lambda: b.close) for b in bars]
    
    ib.cancelHistoricalData(bars)
    
    try:
        return close[0]
    except Exception: 
        return np.nan