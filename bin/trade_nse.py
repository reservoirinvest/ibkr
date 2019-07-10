# trade_nse.py
from z_helper import *

# from json
a = assign_var('nse') + assign_var('common')
for v in a:
    exec(v)
    
ib = get_connected('nse', 'live')

#_____________________________________

