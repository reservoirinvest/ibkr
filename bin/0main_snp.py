# main_snp.py
"""Main program for snp
Date: 23-July-2019
Ver: 1.0
"""

##### Main Program for snp
from z_helper import *

from chains_snp import *
from ohlcs import *
from sized_snp import *
from target_snp import *
from workout_snp import *

# from json
a = assign_var('snp') + assign_var('common')
for v in a:
    exec(v)

def ask_user():
    '''Asks the user for what needs to be done
    Arg: None
    Returns: 0 to 6 int'''
    # Get user input
    askmsg = "\nChoose from the following numbers:\n" + \
            "0) Run ALL (for morning trades)\n" + \
            "1) Chain Generation\n" + \
            "2) OHLCs Generation\n" + \
            "3) Size the options\n" + \
            "4) Target preparation\n"+ \
            "5) Trade in the morning\n" + \
            "6) Workout closing trades (DYNAMIC)\n" + \
            "...Or close window to abort\n\n"

    while True:
        try:
            ip = int(input(askmsg+'\n'))
        except ValueError:
            print("\nSorry, I didn't understand what you entered. Try again!\n")
            continue # Loop again
        if not ip in [0, 1, 2, 3, 4, 5, 6]:
            print(f"\n{ip} is a wrong number! Choose any number from 0 to 7 and press <Enter>\n")
        else:
            break # success and exit loop
    
    return ip

# delete data and log files
def delete_all_data():
    '''Deletes all data and log files
    Arg: None Return: None'''
    folderpaths = ["../data/snp/", "../data/log/"]

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

# do all the functions
def do_all(ib):
    '''Does all the functions outlined
    Args: None
    Returns: None'''
    
    # delete all data and log files
    delete_all_data()
    print("Deleted all data and log files\n")
    
    # do all the functions
    df_chains=get_chains(ib)
    print("Got the chains\n")
    
    df_ohlcs=make_ohlcs(ib, df_chains)
    print("Made the OHLCs\n")
    
    df_sized=sized_snp(ib, df_chains, df_ohlcs)
    print("Sized the options\n")
    
    # Error in margins
    if len(df_sized[df_sized.margin.isnull()]) == len(df_sized):
        print("\nERROR: Margins unavailable. Please run sizing again!\n")
        return None
    
    df_targets = target_snp(ib, df_sized, blacklist)
    print("Build the targets\n")
    
    df_buy = workout_snp(ib)
    
    sell_tb = sells(ib, df_targets, exchange)
    buy_tb = buys(ib, df_buy, exchange)
    place_morning_trades(ib, sell_tb=sell_tb, buy_tb=buy_tb)
    print("Placed the morning trades\n\n")

    return None

#_____________________________________

# userip.py
# the selecting user inputs
if __name__=='__main__':
    userip = ask_user()
    
    with get_connected('snp', 'live') as ib:
        if userip == 0: # Run all
            start = time.time()
            print("\nRunning ALL\n")
            do_all(ib)
            print(f"\nTook {codetime(time.time()-start)} to complete do_all\n")
            
        elif userip == 1: # Chain Generation
            start = time.time()
            print("\nGetting Chains\n")
            df_chains=get_chains(ib)
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
            df_sized=sized_snp(ib, df_chains, df_ohlcs)
            print(f"\nOptions sized in {codetime(time.time()-start)}\n")
            
        elif userip == 4: # Target prepration
            start = time.time()
            print("Preparing targets\n")
            df_chains = pd.read_pickle(fspath+'chains.pkl')
            df_ohlcs = pd.read_pickle(fspath+'ohlcs.pkl')
            df_sized = pd.read_pickle(fspath+'sized_snp.pkl')
            df_targets = target_snp(ib, df_sized, blacklist)
            print(f"\nMade SELL targets in {codetime(time.time()-start)}\n")
            
        elif userip == 5: # Trade in the morning
            start = time.time()
            print("Trading in the morning\n")
            df_chains = pd.read_pickle(fspath+'chains.pkl')
            df_ohlcs = pd.read_pickle(fspath+'ohlcs.pkl')
            df_sized = pd.read_pickle(fspath+'sized_snp.pkl')            
            df_targets = pd.read_pickle(fspath+'targets.pkl')
            df_buy = workout_snp(ib)
            
            sell_tb = sells(ib, df_targets, exchange)
            buy_tb = buys(ib, df_buy, exchange)
            morning_trades = place_morning_trades(ib, sell_tb=sell_tb, buy_tb=buy_tb)
            print(f"\nCompleted morning trades in {codetime(time.time()-start)}\n")
            
        elif userip == 6: # Workout closing trades for new fills
            start = time.time()
            print("Closing new fills\n")
            df_buy = workout_snp(ib)
            buy_tb = buys(ib, df_buy, exchange)
            doTrades(ib, buy_tb)
            print(f"\nFilled close BUY orders in {codetime(time.time()-start)}\n")

#_____________________________________

