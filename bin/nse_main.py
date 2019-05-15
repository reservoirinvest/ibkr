# nse_main.py
# Main program for nse

from helper import *
from nse_func import *

# Do assignments
a = assign_var('nse')
for v in a:
    exec(v)

from ib_insync import *

# Get user input
askmsg = "1) Build Targets (45 mins)\n" + "2) Dynamically Manage (2 mins)\n" + "   Please choose 1 or 2: "
while True:
    try:
        ip = int(input(askmsg))
    except ValueError:
        print("Sorry, I didn't understand that")
        continue # loop again
    if not ip in [1, 2]:
        print("Please choose the right number")
        continue # loop again
    else:
        break # success and exit loop

# do the appropriate function
if ip is 1:    # build the base
    
    with get_connected('nse', 'live') as ib:
        
        with open(logpath+'build.log', 'w'):
            pass # clear the run log
        
        util.logToFile(logpath+'build.log')
        
        util.logging.info('####                 NSE BASE BUILD STARTED                  ####')
        base(ib)
        util.logging.info('____________________NSE BASE BUILD COMPLETE______________________')

        # get the rom and optPrice
        opts(ib)
        util.logging.info('__________________NSE OPTIONS BUILD COMPLETE_____________________')

        # make the targets
        targets(ib)
        util.logging.info('_________________NSE TARGETS BUILD COMPLETED_____________________')
        
        # make the watchlists
        watchlists(ib)
        
else:         # place dynamic trades

    with get_connected('nse', 'live') as ib:

        util.logToFile(logpath+'dynamic.log')

        util.logging.info('####               START of Dynamic Manage                   ####')

        # run the dynamic update
        dynamic(ib)

        util.logging.info('___________________END of Dynamic Manage_________________________')
        
# code put inside this will not be executed if nse_main is imported!
if __name__ == "__main__":
    pass

#_____________________________________

