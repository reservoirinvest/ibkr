# main_nse.py
"""Main program for NSE
Date: 23-July-2019
Ver: 1.0
"""

##### Main Program for NSE
from z_helper import *

from chains_nse import *
from ohlcs import *
from sized_nse import *
from target_nse import *
from workout_nse import *
from x_4Capstocks import *

# from json
a = assign_var('nse') + assign_var('common')
for v in a:
    exec(v)

def ask_user():
    '''Asks the user for what needs to be done
    Arg: None
    Returns: 0 to 7 int'''
    # Get user input
    askmsg = "\nChoose from the following numbers:\n" + \
            "0) Prepare for morning trades\n" + \
            "1) Chain Generation\n" + \
            "2) OHLCs Generation\n" + \
            "3) Size the options\n" + \
            "4) Target preparation\n"+ \
            "5) Place morning trades\n" + \
            "6) Workout closing trades (DYNAMIC)\n" + \
            "7) Zip Capstocks BUYs\n\n" + \
            "...Or close window to abort\n\n"

    while True:
        try:
            ip = int(input(askmsg+'\n'))
        except ValueError:
            print("\nSorry, I didn't understand what you entered. Try again!\n")
            continue # Loop again
        if not ip in [0, 1, 2, 3, 4, 5, 6, 7]:
            print(f"\n{ip} is a wrong number! Choose any number from 0 to 7 and press <Enter>\n")
        else:
            break # success and exit loop
    
    return ip

# delete data and log files
def delete_all_data():
    '''Deletes all data and log files
    Arg: None Return: None'''
    folderpaths = ["../data/nse/", "../data/log/"]

    for folder in folderpaths:
        for files in listdir(folder):
            file_path = path.join(folder, files)
            try:
                if path.isfile(file_path):
                    unlink(file_path)
            except Exception as e:
                print(e)
                
    return None

# generate ohlcs
def make_ohlcs(ib, df_chains):
    '''Makes OHLCs for the underlying symbols
    Args:
        (ib) as connection object
        (df_chains) as DataFrame chain
    Returns: df_ohlcs as DataFrame of ohlcs and pickles them'''
    id_sym = df_chains.set_index('undId').symbol.to_dict()

    df_ohlcs = ohlcs(ib, id_sym, fspath, logpath)

    df_ohlcs.to_pickle(fspath+'ohlcs.pkl')
    
    return df_ohlcs

# get CAPSTOCK trades
def get_capstocks(cap_blacklist):
    '''Prepares list of Capstock Trades
    Arg: (cap_blacklist) as list of symbols with positions that need to be excluded
    Returns: DataFrame of capstock trades. Also makes a spreadsheet and watchlist'''
    
    df_cap=capstocks(cap_blacklist)
    
    return df_cap

# do all the functions
def do_all(ib):
    '''Does all the functions outlined
    Args: None
    Returns: None'''
    
    # delete all data and log files
    delete_all_data()
    print("Deleted all data and log files\n")
    
    # do all the functions
    df_chains=get_chains(nseweb=False)
    print("Got the chains\n")
    
    df_ohlcs=make_ohlcs(ib, df_chains)
    print("Made the OHLCs\n")
    
    df_sized=sized_nse(ib, df_chains, df_ohlcs)
    print("Sized the options\n")
    
    # Error in margins
    if len(df_sized[df_sized.margin.isnull()]) == len(df_sized):
        print("\nERROR: Margins unavailable. Please run sizing again!\n")
        return None
    
    df_targets = target_nse(ib, df_sized, blacklist, nse_assignment_limit)
    print("Build the targets\n")
    
    df_sell = df_targets
    df_buy = workout_nse(ib)

    return df_buy, df_sell

#_____________________________________

# userip.py
# the selecting user inputs
if __name__=='__main__':
    userip = ask_user()
    
    with get_connected('nse', 'live') as ib:
        if userip == 0: # Run all
            start = time.time()
            print("\nRunning ALL\n")
            df_buy, df_sell = do_all(ib)
            print(f"\nTook {codetime(time.time()-start)} to complete do_all\n")
            
        elif userip == 1: # Chain Generation
            start = time.time()
            print("\nGetting Chains\n")
            df_chains=get_chains(nseweb=False)
            print(f"\nGot option chains in {codetime(time.time()-start)}\n")
            
        elif userip == 2: # OHLC Generation
            start = time.time()
            print("\nGenerating OHLCs\n")
            df_chains = pd.read_pickle(fspath+'chains.pkl')
            df_ohlcs=make_ohlcs(ib, df_chains)
            print(f"\nOHLCs generated in {codetime(time.time()-start)}\n")
            
        elif userip == 3: # Size the options
            start = time.time()
            print("Sizing the options\n")
            df_chains = pd.read_pickle(fspath+'chains.pkl')
            df_ohlcs = pd.read_pickle(fspath+'ohlcs.pkl')
            df_sized=sized_nse(ib, df_chains, df_ohlcs)
            print(f"\nOptions sized in {codetime(time.time()-start)}\n")
            
        elif userip == 4: # Target prepration
            start = time.time()
            print("Preparing targets\n")
            df_chains = pd.read_pickle(fspath+'chains.pkl')
            df_ohlcs = pd.read_pickle(fspath+'ohlcs.pkl')
            df_sized = pd.read_pickle(fspath+'sized_nse.pkl')
            df_targets = target_nse(ib, df_sized, blacklist, nse_assignment_limit)
            print(f"\nMade SELL targets in {codetime(time.time()-start)}\n")
            
        elif userip == 5: # Place morning trades
            start = time.time()
            print("Trading in the morning\n")
            df_chains = pd.read_pickle(fspath+'chains.pkl')
            df_ohlcs = pd.read_pickle(fspath+'ohlcs.pkl')
            df_sized = pd.read_pickle(fspath+'sized_nse.pkl')            
            df_targets = pd.read_pickle(fspath+'targets.pkl')

            # cancel all existing trades
            ib.reqGlobalCancel() 
            
            df_buy = workout_nse(ib)
            buy_tb = buys(ib, df_buy, exchange)
            
            sell_tb = sells(ib, df_targets, exchange)

            # place the morning trades            
            morning_trades = place_morning_trades(ib, sell_tb=sell_tb, buy_tb=buy_tb)
            print(f"\nCompleted morning trades in {codetime(time.time()-start)}\n")
            
        elif userip == 6: # Workout closing trades for new fills
            start = time.time()
            print("Closing new fills\n")
            
            df_buy = workout_nse(ib)
            
            if df_buy.empty:
                print("\nNothing to buy!\n")
            else:
                buy_tb = buys(ib, df_buy, exchange)
                doTrades(ib, buy_tb)
                print(f"\nFilled close BUY orders in {codetime(time.time()-start)}\n")
            
        elif userip == 7: # Capstocks BUY generation
            print("Generating BUYs for Capstocks\n")
#             cap_blacklist = 'ASHOKLEY,BHARATFORG,GRASIM,PETRONET,SUNTV,ICICIPRULI,ARVIND,BSOFT,ENGINERSIN,TATAELXSI,EICHERMOT,MOTHERSUMI,TVSMOTOR'.split(',')
            get_capstocks(cap_blacklist=cap_blacklist)
            print("Completed generating Capstocks BUY list\n")

#_____________________________________

