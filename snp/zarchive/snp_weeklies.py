# Gets a list of NYSE Weekly tickers

import pandas as pd
import numpy as np
import re
import datetime


def get_nyse_weeklies():
    """Makes a dataframe of NYSE tickers with all the weekly expiries
    Arg: None
    Returns: DataFrame of nyse weeklies"""

    # Download cboe weeklies to a dataframe
    dls = "http://www.cboe.com/publish/weelkysmf/weeklysmf.xls"
    raw_df = pd.read_excel(dls)

    # Extract expiration dates by type
    df1 = raw_df.iloc[:9, 2:]

    df2 = df1.T

    # Types of expiry columns
    expiry_type = ['Std_Week', 'Expanded_Week', 'EOW', 'SPX_EOW', 'XSP_Wed', 'Mon_Wed', 'Mon', 'Wed', 'VIX']
    df2.columns = expiry_type

    # Drop the first row
    df2.drop('Unnamed: 2', inplace=True)

    # Make an expiries dataframe with a new index
    expiries = df2.reset_index().drop('index', 1)

    # Extract Tickers and the Expiry Markers [x]

    first_row = 12

    # Select only columns of interest
    a = raw_df.iloc[first_row:, 0]
    b = raw_df.iloc[first_row:, 2:4]
    c = raw_df.iloc[first_row:, 5:12]
    tickers = pd.concat((pd.DataFrame(a), b, c), 1)

    # Rename columns
    col_names = ['Ticker', 'Desc', 'Type'] + expiry_type[:3] + expiry_type[5:9]
    tickers.columns = col_names

    # Remove non-null tickers
    tickers = tickers.loc[tickers.Ticker.notnull(), :]

    # Remove strange characters from tickers
    pattern = re.compile('[\W_]+')  # Matches any non-word character and underscore
    pattern.sub('', tickers.Ticker[14])
    tickers.Ticker = tickers.Ticker.apply(lambda x: pattern.sub('', x))

    # Clean up to reflect Index Funds in Type
    tickers.loc[tickers.Type.str.contains('Index', na=False), 'Type'] = 'Index'

    # Get target columns for each ticker
    scriplist = []

    for target_col in list(tickers)[3:]:
        ticker_cols = ['Ticker', 'Desc', 'Type']

        # Gets the tickers for target_col
        tick = tickers.loc[(tickers[target_col]) == 'X', ticker_cols]

        # Get the cleaned expiries
        exp = expiries.loc[:, target_col]
        exp = pd.to_datetime(exp, errors='coerce').dropna()

        # Repeat Expiries for the ticker
        scrips = pd.DataFrame(np.repeat(tick.values, len(exp), axis=0),
                              columns=tick.columns,
                              index=np.tile(exp, len(tick))).rename_axis('Expiry').reset_index()

        # Appends the scrips for each expiry type
        scriplist.append(scrips)

    # Drops duplicates and makes a neat dataframe!
    scriplist = pd.concat(scriplist).drop_duplicates(keep='first').reset_index(drop=True)

    return scriplist


def main():
    """Assembles NYSE weeklies in S&P500
    Arg: None
    Returns: Dataframe with S&P500 weeklies with expiry and DTE
    """
    df = get_nyse_weeklies()  # Get the weeklies

    # Get SnP 100
    url = "https://en.wikipedia.org/wiki/S%26P_100"
    snp100 = pd.read_html(url, header=0)[2]
    snp100.head()

    snp_weeklies = snp100.merge(df, left_on='Symbol', right_on='Ticker').drop(['Symbol', 'Name'], 1)

    # Get the non-Equities
    df1 = df[df.Type != 'Equity']

    # Merge the two dataframes
    df2 = pd.concat([snp_weeklies, df1], axis=0)

    # Get the DTE
    df2['DTE'] = (df2.Expiry - datetime.datetime.now()).dt.days
    df2.Expiry = df2.Expiry.dt.strftime('%Y%m%d')

    return df2


# if this script is executed then main will be executed
if __name__ == '__main__':
    main()