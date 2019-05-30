# nse_main.py
# Main program for nse

from helper import *
from nse_func import *

# Do assignments
a = assign_var('nse')
for v in a:
    exec(v)

from ib_insync import *

# logic to create opts.pickle
fs = [f for f in listdir(fspath) if f[-3:] == 'opt'] # opt pickles files

if fs: # if the file list is not empty
    df_opts = pd.concat([pd.read_pickle(fspath+f) for f in fs]).reset_index(drop=True)

    # write to pickle
    df_opts.to_pickle(fspath+'opts.pickle')

# Get user input
askmsg = "1) Build ALL Opts+Target\n" + "2) Build remaining Opts+Target\n" + "3) Dynamically Manage (2 mins)\n\n" + "   Please choose 1, 2 or 3:\n\n"
while True:
    try:
        ip = int(input(askmsg))
    except ValueError:
        print("Sorry, I didn't understand that")
        continue # loop again
    if not ip in [1, 2, 3]:
        print("Please choose the right number")
        continue # loop again
    else:
        break # success and exit loop

# do the appropriate function
if ip in [1, 2]:    #  build ALL the options
    
    with get_connected('nse', 'live') as ib:
        
        with open(logpath+'build.log', 'w'):
            pass # clear the run log
        
        util.logToFile(logpath+'build.log')
        
        util.logging.info('####                     NSE BUILD STARTED                  ####')
        print(f'NSE options build started at {datetime.datetime.now()}...')
        s = time.perf_counter()
        
        # generate the symexplots
        df_l = symexplots(ib)
        
        # get the series of rows for options
        rows = [s for i, s in df_l.iterrows()]
        
        if ip is 2:
            # remove rows already pickled
            pkl_done = pd.read_pickle(fspath+'opts.pickle').symbol.unique()
            rows = [row for row in rows if row.symbol not in pkl_done]

        # get the options
        tqr = trange(len(rows), desc='Processing', leave=True) # Initializing tqdm # initializing tqdm
        opts = [] # initializing list of opts

        for row in rows:
            tqr.set_description(f"Processing [{row.symbol}]")
            tqr.refresh() # to show immediately the update
            opts.append([do_an_opt(ib, row, df_l)])
            tqr.update(1)
        tqr.close()

#         [do_an_opt(ib, row, df_l) for row in rows]

        # make the targets
        targets(ib)

        elapsed = (time.perf_counter() - s)/60

        util.logging.info('________________________NSE BUILD COMPLETE______________________')
        print(f'NSE options build completed at {datetime.datetime.now()}\n')
        print(f'...executed in {elapsed:0.1f} minutes.')

          
elif ip is 3:         # place dynamic trades

    with get_connected('nse', 'live') as ib:

        # run the dynamic update
        dynamic(ib)
        
# code put inside this will not be executed if nse_main is imported!
if __name__ == "__main__":
    pass

#_____________________________________

