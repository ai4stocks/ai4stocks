# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:21:36 2017

@author: docul
"""

import pandas as pd
import tushare as ts
import stockbasic as sb
from matplotlib.pylab import date2num
import datetime

import numpy as np
import tushare as ts
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.finance as mpf

# Matplotlib 显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']  #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  #解决保存图像是负号'-'显示为方块的问题

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
        
        # convert 'date' to matplotlib number
        onestock['mpl.date'] = date_to_num(onestock['date'].values)
    
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


def test_sync():
    print('exits 2.0')

#
# generate_samples
#

# number of bars to look back to form a sample
CONST_LOOKBACK_SAMPLES = 120

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
    X_frames = []
    Y_frames = []
    
    # debug
    nb_samples = 0
    verbose = False
    verbose_l1 = False

    for index, row in stock_data.iterrows():
        # debug
        # if (row['mpl.date'] >= 735218.0) and (row['mpl.date'] <= 735277.0):
        #     verbose = True
        # else:
        #     verbose = False
        
        # is this row a buy-point?
        if not sb.is_squeeze_buy_point(stock_data, index):
            if verbose:
                print(index, ' is NOT squeeze buypoint')
            # check next row
            continue

        # Yes, it is a buy-point.
        if verbose:
            print(index, ' is squeeze buypoint')
    
        # Let's check when is the sell-point.
        sell_index, sell_reason = sb.get_sell_point(stock_data, index)

        if verbose:
            print('sellpoint: ', sell_index, ' reason: ', sell_reason)

        # Do we hit a sell point?
        if sell_reason == 0: # No, skip it
            print('No sell_point for. Skip ', index)
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
            print('First location < 0, Skip ', index)
            continue

        # Slicing. These are totally CONST_LOOKBACK_SAMPLES of records.
        x_sample = stock_data.iloc[first_location:(location_of_buy_point + 1)]
        # x_sample Validity check.
        if x_sample.isnull().values.any():
            # skip it
            print('Null values in x_sample. Skip. ', index)
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
            print('Null value in y_smaple. skip. ', index, sell_index)
            continue

        if verbose:
            print('append x_sample, y_smaple to X/Y_frames')
            print('y_sample: ', y_sample)
            
        if verbose_l1:
            print(nb_samples, ' y_sample: ', index, sell_index)
        
        # Add into X/Y_frames
        X_frames.append(x_sample)
        Y_frames.append(y_sample)
        nb_samples += 1

    #### END of for loop ####

    # Add N-record to X_all
    X_all = pd.concat(X_frames, ignore_index = False)
    # Add sell-point information to Y_all.
    Y_all = pd.concat(Y_frames, axis = 1)
    Y_all = Y_all.T
    
    if verbose_l1:
        print('number of samples for ', index, ': ', nb_samples)
        print('shape of X/Y:', X_all.shape, Y_all.shape)
    
    return X_all, Y_all


#
# date_to_num
#


def date_to_num(dates):
    ''' Function to convert tushare 'date' string to matplotlib datenum
    
    Input
    =====
    dates: ndarray of tushare 'date' strings. Eg. ['2013-01-31', ...]
    
    Output
    ======
    Return: list of float datetime value compatible to matplotlib: floating point 
            numbers which represent time in days since 0001-01-01 UTC, plus 1. 
            For example, 0001-01-01, 06:00 is 1.25, not 0.25.
    
    Example
    =======
        stock_data['mpl.date'] = date_to_num(stock_data['date'].values)

    '''
    num_time = []
    for date in dates:
        date_time = datetime.datetime.strptime(date,'%Y-%m-%d')
        num_date = date2num(date_time)
        '''
            matplotlib.dates.date2num(d)
                Converts datetime objects to Matplotlib dates
        '''
        num_time.append(num_date)
    return num_time


#
# plog_stock_data
#


def plot_stock_data(stock_data, title_postfix=''):
    ''' Function to plot stock_data
    
    Input
    =====
    stock_data: DataFrame, with columns 'date', 'code', 'close', 'volumn', etc.
    
    Output
    ======
    Return: None
    '''
   
    # make a local copy
    sdata = stock_data.copy(deep=False)
    # convert index 'date' to a column
    sdata.reset_index(level=1, inplace=True)
    
    # convert date to num
    sdata['mpl.date'] = date_to_num(sdata['date'].values)

    fig, axes = plt.subplots(7, sharex=True, figsize=(15,14),
                             gridspec_kw={'height_ratios':[3,1,1,1,1,1,1]})
    
    # axes[0]: k-line
    mpf.candlestick_ochl(axes[0],
                         sdata[['mpl.date', 'open', 'close', 'high', 'low']].values,
                         width=1.0,
                         colorup = 'g',
                         colordown = 'r')
    # axes[0]: EMA8, EMA21
    axes[0].plot(sdata['mpl.date'].values, sdata['EMA8'].values, 'm', label='EMA8')
    axes[0].plot(sdata['mpl.date'].values, sdata['EMA21'].values, 'c', label='EMA21')
    axes[0].legend(loc=0)
    axes[0].grid(True)
    
    axes[0].set_title(stock_data['code'].iloc[0] + ' ' + title_postfix)
    axes[0].set_ylabel('Price')
    axes[0].grid(True)
    axes[0].xaxis_date()

    # axes[1]: volume
    axes[1].bar(sdata['mpl.date'].values-0.25, sdata['volume'].values, width= 0.5)
    axes[1].set_ylabel('Volume')
    axes[1].grid(True)
    
    # axes[2]: MTMMA
    bars = axes[2].bar(sdata['mpl.date'].values-0.25, sdata['MTMMA'].values, width=0.8)
    for bar in bars:
        if bar.get_height() > 0:
            bar.set_color('g')
        else:
            bar.set_color('r')
        
    # axes[2]: SQUEEZE
    axes[2].plot(sdata['mpl.date'].values-0.25,
                 [0 if x == sb.CONST_SQUEEZE_ONGOING else float('nan') for x in sdata['SQUEEZE'].values], # rescale
                 'ko',
                 label='SQUEEZE')
    axes[2].set_ylabel('SQZ')
    axes[2].grid(True)
    
    # axes[3]: TTM WAVE C
    bars = axes[3].bar(sdata['mpl.date'].values-0.25, sdata['MACD6'].values, color='red', width=0.8, alpha=0.8)
    bars = axes[3].bar(sdata['mpl.date'].values-0.25, sdata['HIST5'].values, color='orange', width=0.8, alpha=0.8)
    axes[3].set_ylabel('WAVE C')
    axes[3].grid(True)
   
    # axes[4]: TTM WAVE B
    bars = axes[4].bar(sdata['mpl.date'].values-0.25, sdata['HIST4'].values, color='magenta', width=0.8, alpha=0.8)
    bars = axes[4].bar(sdata['mpl.date'].values-0.25, sdata['HIST3'].values, color='teal', width=0.8, alpha=0.8)
    axes[4].set_ylabel('WAVE B')
    axes[4].grid(True)
   
    # axes[5]: TTM WAVE A
    bars = axes[5].bar(sdata['mpl.date'].values-0.25, sdata['HIST2'].values, color='lawngreen', width=0.8, alpha=0.8)
    bars = axes[5].bar(sdata['mpl.date'].values-0.25, sdata['HIST1'].values, color='yellow', width=0.8, alpha=0.8)
    axes[5].set_ylabel('WAVE A')
    axes[5].grid(True)
    
    # axes[6]: ADX
    axes[6].plot(sdata['mpl.date'].values, sdata['ADX'].values, 'm', label='ADX')
    axes[6].set_ylabel('ADX')
    axes[6].grid(True)
   
    return
    
