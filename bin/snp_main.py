# snp_main.py
# Main program for snp

from ib_insync import *

from helper import *
from snp_func import *

if __name__ == '__main__':
    with get_connected('snp', 'live') as ib:
        snp_weekend_process(ib)
    with get_connected('snp', 'live') as ib:
        snp_everyday_process(ib)

#_____________________________________

