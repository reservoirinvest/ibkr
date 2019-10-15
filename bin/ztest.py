# imports.py
from support import *

import time
from datetime import datetime
from itertools import product
from math import erf, sqrt
import sys

# start the ib loop

# connect to the market set in variables.json
try:
    if not ib.isConnected():
        ib = get_connected(market, 'live')
except Exception as e:
    ib = get_connected(market, 'live')

# from json
a = assign_var(market) + assign_var('common')
for v in a:
    exec(v)

# reset the log
with open(logpath+'test.log', 'w'):
    pass # clear the run log

util.logToFile(logpath+'test.log')
    
jup_disp_adjust() # adjust jupyter's display

#_____________________________________

