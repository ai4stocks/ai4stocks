# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:21:36 2017

@author: docul
"""

import pandas as pd
import tushare as ts
import stockbasic as sb

#
# printProgressBar
# Copied from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
# 

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '+'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


#
# 读取指定股票集的数据
# fetch_raw_data
#

def fetch_raw_data(stock_list, start_date, end_date):
    ''' Function to fetch stocks' raw OHLC and volume data
    
    Input
    =====
    stock_list: a DataFrame
        with column 'code', each row stands for a stock ID.
    start_date: String
        start date, in format 'YYYY-MM-DD'
    end_date: String 
        end date, in format 'YYYY-MM-DD'
    
    Output
    ======
    Return: MultiIndex DataFrame
        indexed by 'code' and 'date'
    
    Example
    =======
    >>> import tushare as ts
    >>> # stock_list_hs300 = ts.get_hs300s()
    >>> stock_list_test = pd.DataFrame([{'code':'000001'}, {'code':'000002'}])
    >>> print(stock_list_test)
    >>> a = fetch_raw_data(stock_list_test, '2010-01-01', '2017-12-20')
    '''
    all_data = pd.DataFrame()
    total = stock_list.shape[0]
    i = 0
    
    # Initial call to print 0% progress
    printProgressBar(0, total, prefix = 'Progress:', suffix = 'Complete', length = 60)

    # go through each stock in the stock_lists
    # iterate the list
    for index, row in stock_list.iterrows():
        i += 1
        
        # fetch K data of each code
        onestock = ts.get_k_data(row['code'], start=start_date, end=end_date)
        # set index to 'date'
        onestock.set_index('date', inplace=True)
        # print('this is '+row['code'])
        # print(onestock.head())
        # pack it into one DataFrame
        all_data = all_data.append(onestock)
        printProgressBar(i, total, prefix = 'Fetching Progress:', suffix = 'Complete', length = 60)

    # set multiIndex, by 'code', and by 'date'
    all_data.set_index([all_data['code'], all_data.index], drop=True, inplace=True)
    
    return all_data




#
# generate_samples
#

# number of bars to look back to form a sample
CONST_LOOKBACK_SAMPLES = 60

def generate_samples(stock_data):
    ''' Function to generate samples. 选出符合规则的数据做训练用例.
        从给定的股票数据，根据 买入规则， 卖出规则，选出相符合的数据序列，用作 训练用例
   
    Explain
    ======= 
    Given a stock_data, generate buy/sell samples.
        Currently, using is_squeeze_buy_point() to find buy points. And using get_sell_point() to find sell points

    Input
    =====
    stock_data: DataFrame
        stock_data with full features 'SQZ', WAVE C/B/A, ATR, ADX, etc.

    Output
    ======
    X_all: DataFrame
        all samples' X part, concatenated, in DataFrame
    Y_all: DataFrame
        all samples' Y part, concatenated, in DataFrame
    '''
    # Initialize X_all and Y_all
    X_all = pd.DataFrame()
    Y_all = pd.DataFrame()

    for index, row in stock_data.iterrows():
        # is this row a buy-point?
        if not sb.is_squeeze_buy_point(stock_data, index):
            ## print(index + ' is NOT')
            # check next row
            continue

        # Yes, it is a buy-point.
        # Let's check when is the sell-point.
        sell_index, sell_reason = sb.get_sell_point(stock_data, index)

        # Do we hit a sell point?
        if sell_reason == 0: # No, skip it
            continue

        ## print('=======')
        ## print(index + ' is YES')
        ## print(sell_index, 'is SELL POINT')
        ## print(sell_reason, 'is SELL REASON')
        ## print('Buy  @ ', stock_data['close'][index])
        ## print('Sell @ ', stock_data['close'][sell_index])
        ## print('=======')

        # Back fetch N-record
        location_of_buy_point = stock_data.index.get_loc(index)
        first_location = location_of_buy_point - CONST_LOOKBACK_SAMPLES + 1
        if first_location < 0: # there is no enough records to form a valid sample
            # skip it
            continue

        # Slicing. These are totally CONST_LOOKBACK_SAMPLES of records.
        x_sample = stock_data.iloc[first_location:(location_of_buy_point + 1)]
        # x_sample Validity check.
        if x_sample.isnull().values.any():
            # skip it
            continue

        # create y_sample as a pandas.Series
        y_raw_data = {'code': row['code'],
                      'buy_date': index,
                      'buy_price': stock_data['close'][index],
                      'sell_date': sell_index,
                      'sell_price': stock_data['close'][sell_index],
                      'sell_reason': sell_reason}
        y_sample = pd.Series(y_raw_data)
        # y_sample validity check
        if y_sample.isnull().values.any():
            # skip it
            continue

        # Add N-record to X_all
        X_all = X_all.append(x_sample, ignore_index = False)
        # Add sell-point information to Y_all.
        Y_all = Y_all.append(y_sample, ignore_index = True)

        #### END of for loop ####

    return X_all, Y_all