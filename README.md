# Introduction

* This set of programs attempts to set up a dynamic framework in IB for two markets:
 - USA (SNP)
 - India (NSE)

# Concept and Structure
The overall concept and structure is as below:
![Alt structure](./pic/structure.svg?sanitize=true "Overall Structure")

# Requirements

The programs require the following environment set up 
 - for [python](https://www.python.org/downloads/)
 - using [Jupyter Notebook](http://jupyter.org/install) as IDE
    - [Anaconda distribution](https://www.anaconda.com/distribution/) can be used instead of raw jupyter.
 - to run [IBKR TWS API]( https://interactivebrokers.github.io/) in C:\ root for Windows
 - on [TWS](https://www.interactivebrokers.com.hk/en/index.php?f=16042) or [IB Gateway](https://www.interactivebrokers.com.hk/en/index.php?f=16457)
 - with [IB_insync](https://rawgit.com/erdewit/ib_insync/master/docs/html/readme.html#) library
 - backed up into [git](https://git-scm.com/downloads)
 
 **Note:** The instructions are for Windows 10. The same process can be used for other OS - such as Ubuntu and MacOS.
 
## Setup

* This section is borrowed from [ib_insync](https://rawgit.com/erdewit/ib_insync/master/docs/html/readme.html)

  1. Install [python](https://www.python.org/downloads/) latest release (3.7.x at the time of this writing)
  
  
  2. Install ib_insyc with the command: 
  > *pip3 install -U ib_insync*
  
  
  3. Install [Interactive Brokers Python API](http://interactivebrokers.github.io/) - latest version
     * IBKR's TWS API should be in the root folder. This needs to be shared between Python and Jupyter.
    
    
  4. Install jupyter and its dependencies with the following command:
  > *pip3 install -U jupyter numpy pandas requests lxml html5lib BeautifulSoup4 quart*
  
 - Some packages like [pywinpty](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pywinpty) in Windows may fail during installation of jupyter in command line interface. There are two options to resolve this:
    - Manually install the packages from [Christoph Gohlke's Windows Binary for PEP](https://www.lfd.uci.edu/~gohlke/pythonlibs/)
	- Use [Anaconda distribution](https://www.anaconda.com/distribution/)
  
     for re-installing / upgrade the command is:
  > *pip3 install --force-reinstall --upgrade jupyter*
      
      
  5. To get Jupyter recognize TWS API, go to the *C:\TWS API\source\pythonclient* folder and run *python setup.py install*
    
    
  6. Set the API for TWS / IB Gateway to the appropriate _Socket Port_
     * For our example we will use IB Gateway's paper trading account with a Socket Port of 4002
    
  
  7. Make a project folder and set up [Git](https://git-scm.com/)
    For a quick understanding of most important git commands, refer [this git guide](http://rogerdudler.github.io/git-guide/)
  
# Programs

Programs are stored in the _bin_ sub-directory. 
 - Those ending with *.ipynb* are jupyter files.
 - And those ending with *.py* are python files.

Here is a brief about them:

  1. **main** - contains the main program, that gives multiple options.
  2. **support** - contains programs helping out the main. It has all the related modules.
  3. **jup2py.ipynb** - is a python program in jupyter that converts all the jupyter files (such as main.ipynb and support.ipynb) to python (.py) modules.
  4. **variables.json** - is a json file that stores parameters to be adjusted. It has *common* and *market-specific* parameters.

## The "main" program

The main program provides options to:
1. Freshly create *target.pkl*, *writecovers.pkl* and other pickle file. 
2. Place the freshly created trades, closures and covers.

The above is to be run once, before the market opens.

3. Place closures only. This can be run any time.

## The "support" program

The support programs contain two types of modules.

### Core functions
Performing functions such as 
 - getting the option chains, 
 - ohlcs, 
 - sizing the options based on variables.json parameters, 
 - building targets (SELLs), closure (BUYs) and covers (for assigned long and short stocks).
 - setting 'harvest' prices
 - getting remaining quantities
 
### Helper functions
For activities such as:
 - error catcher in list comprehension
 - setting right precision for markets (nse: 0.05, snp: 0.01)
 - placing and cancelling trades
 
 # Data
 
 Data is stored in its own folder in the following structure:
  - *log* containing log files
  - *[market-name]* containing market specific pickles. for e.g. <snp> | <nse>
 
 ---
 
 # To-do
 
 1. Convert programs to modular asyncio
 
 2. Make a trading dashboard with graphical UI. Probable GUI options are:
    with quart + Dash
	with EEL
	with pyQt5

 3. Create self-installing package
 
 4. Port the solution to cloud (AWS/Azure)
 
 5. Create mobile app
 
## Functional improvements / study laundry list
* Function to adjust n-band to remaining quantities
* Function for NLV, Initial Margin and Composite ROI computation
* Function for naked mean and median RoM and PoP from target
* Function for Event triggered 'dynamic' workout
* Make a json list of scrips with physical delivery for NSE
* Function to adjust ROI and POP based on expiry date
* From OHLCs - use Bollinger Bands + Keltner Channel + TRIX + Ultimate Oscillator
* Using FLEX to analyze historical trading patterns