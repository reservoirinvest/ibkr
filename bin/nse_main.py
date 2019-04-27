# nse_main.py
# Main program for nse

from ib_insync import *

from helper import *
from nse_func import *

if __name__ == '__main__':
    with get_connected('nse', 'live') as ib:
        nse_weekend_process(ib) # generates lot margin and option pickles
    with get_connected('nse', 'live') as ib:
        nse_everyday_process(ib) # generates target list after evlaluation

#_____________________________________

