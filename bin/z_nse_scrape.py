# x.py
from helper import *
from nse_func import *


# Do assignments
a = assign_var('nse')
for v in a:
    exec(v)

from ib_insync import *

# ib =  get_connected('nse', 'live')

with open(logpath+'ztest.log', 'w'):
    pass # clear the run log

util.logToFile(logpath+'ztest.log')

#_____________________________________

