# Introduction

* This set of programs do semi-automated trading with IBKR in two markets
 - USA (SNP)
 - India (NSE)

# Concept and Structure
The overall concept and structure is as below:
![Alt structure](./pic/structure.svg?sanitize=true "Overall Structure")

# Pre-requisites

The programs require the following environment set up 
 - for [python](https://www.python.org/downloads/)
 - using [Jupyter Notebook](http://jupyter.org/install) as IDE
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
    
    
  4. Install jupyter+numpy+pandas with the command:
  > *pip3 install -U jupyter numpy pandas requests lxml html5lib BeautifulSoup4*
  
     for re-installing / upgrade the command is:
  > *pip3 install --force-reinstall --upgrade jupyter*
      
      
  5. To get Jupyter recognize TWS API, go to the *C:\TWS API\source\pythonclient* folder and run *python setup.py install*
    
    
  6. Set the API for TWS / IB Gateway to the appropriate _Socket Port_
     * For our example we will use IB Gateway's paper trading account with a Socket Port of 4002
    
  
  7. Make a project folder and set up [Git](http://rogerdudler.github.io/git-guide/)
  
# Programs

Programs are stored in the _bin_ sub-directory

## Market-specific programs

Market specific program files have been structured on the following lines with some alphabetical significance:
 - 0main   - the main program
 - chains  - get the option chains
 - ohlcs   - get the open, high, low, close history for 365 days
 - sized   - size the chains to standard deviation rule for writing puts (little lenient) and calls (strict)
 - target  - make target deals with expected price for shorting options
 - workout - for closing the trades filled based on a 'harvest' price that parabollically reduces based on days-to-expiry
 
## Helper programs
 - z_helper - which has common utilities across all markets
 - jup2py - a script to be run in Jupyter that converts jupyter .ipynb files to .py files
 
## JSON

A JSON file has been created to allow interactivity (e.g. throug a GUI) and control. There is currently just one file:
 - variables.json - containing limits (like minimum expected return-on-margin, paths, standard deviation limits, etc.)
 
 
