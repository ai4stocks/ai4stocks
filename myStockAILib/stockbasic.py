# -*- coding: utf-8 -*-

import talib
import pandas as pd
import numpy as np


def ttm_propulsion(stock_data):
    ''' to calculate TTM propulsion indicators

    Explain
    =======
    Function to calculate EMA8 and EMA21

    Input
    =====
    stock_data: pandas.DataFrame
        DataFrame with OHLC, 'volume' and 'code' in each row

    Returns
    =======
    Output: pandas.DataFrame
        DataFrame with 'EMA8', 'EMA21' in each row, use the same index from
        Input stock_data
    '''

    # create a new empty output dataframe, sharing the same index
    newdf = pd.DataFrame(index=stock_data.index.copy())

    # calculate EMA8 and EMA21
    newdf['EMA8'] = talib.EMA(stock_data['close'].values, timeperiod=8)
    newdf['EMA21'] = talib.EMA(stock_data['close'].values, timeperiod=21)

    return newdf


# define squeeze CONST
CONST_SQUEEZE_RELEASED = -1
CONST_SQUEEZE_ONGOING = 1


def ttm_squeeze(stock_data, MULTKC = 1.5, MULT = 1.5, LENGTHKC = 20, LENGTHBB = 20, LENGTHMOM = 12):
    ''' Function to calculate SQUEEZE indicators

    Explain
    =======
    calculate TTM squeeze

    Input
    =====
    stock_data: DataFrame
        with OHLC, 'volume' and 'code' in each row

    Output
    ======
    Return: DataFrame
        with 'SQUEEZE', 'MTMMA' in each row, use the same index from
        Input stock_data
    '''

    # create a new empty output dataframe, sharing the same index
    newdf = pd.DataFrame(index=stock_data.index.copy())

    # calculate ATR
    ATR = talib.ATR(stock_data['high'].values,
                 stock_data['low'].values,
                 stock_data['close'].values,
                 timeperiod=LENGTHKC)  # instead of default 14
    newdf['ATR'] = ATR

    # calculate Keltner Channel
    UPPERKC = talib.MA(stock_data['close'].values, timeperiod=LENGTHKC) + ATR * MULTKC;
    LOWERKC = talib.MA(stock_data['close'].values, timeperiod=LENGTHKC) - ATR * MULTKC;

    newdf['UPPERKC'] = UPPERKC
    newdf['LOWERKC'] = LOWERKC

    # calculate Bolling Band
    UPPERBB, BOLL, LOWERBB = talib.BBANDS(stock_data['close'].values,
                                          timeperiod=LENGTHBB,
                                          nbdevup=MULT,
                                          nbdevdn=MULT)

    newdf['UPPERBB'] = UPPERBB
    newdf['LOWERBB'] = LOWERBB

    squeeze_true = (newdf['LOWERBB'] > newdf['LOWERKC']) & (newdf['UPPERBB'] < newdf['UPPERKC'])
    newdf['SQUEEZE'] = [CONST_SQUEEZE_ONGOING if x else CONST_SQUEEZE_RELEASED for x in squeeze_true]

    MTM = stock_data['close'] - stock_data['close'].shift(LENGTHMOM)
    MTMMA = talib.MA(MTM.values, timeperiod=LENGTHMOM)
    newdf['MTMMA'] = MTMMA

    # drop unwanted columns
    newdf.drop(columns=['ATR', 'UPPERKC', 'LOWERKC', 'UPPERBB', 'LOWERBB'], axis=1, inplace=True)

    return newdf


#
# ttm_wave
#


def ttm_wave(stock_data, SHORT = 8, MID_A = 34, LONG_A = 55,
                                    MID_B = 89, LONG_B = 144,
                                    MID_C = 233, LONG_C = 377):
    ''' Function to calculate TTM WAVE A/B/C

    Explain
    =======
    Calculate TTM Wave A, B and C.

    Input
    =====
    stock_data: DataFrame
        with OHLC, 'volume' and 'code' in each row

    Output
    ======
    Return: DataFrame
        with 'HIST1' to 'HIST5', plus 'MACD6' in each row, use the same index
        from Input stock_data
    '''

    # create a new empty output dataframe, sharing the same index
    newdf = pd.DataFrame(index=stock_data.index.copy())

    # calculate HIST1
    FASTMA1 = talib.EMA(stock_data['close'].values, timeperiod=SHORT)
    SLOWMA1 = talib.EMA(stock_data['close'].values, timeperiod=MID_A)
    MACD1 = FASTMA1 - SLOWMA1
    SIGNAL1 = talib.EMA(MACD1, timeperiod=MID_A)
    HIST1 = MACD1 - SIGNAL1

    newdf['HIST1'] = HIST1

    # calculate HIST2
    FASTMA2 = talib.EMA(stock_data['close'].values, timeperiod=SHORT)
    SLOWMA2 = talib.EMA(stock_data['close'].values, timeperiod=LONG_A)
    MACD2 = FASTMA2 - SLOWMA2
    SIGNAL2 = talib.EMA(MACD2, timeperiod=LONG_A)
    HIST2 = MACD2 - SIGNAL2

    newdf['HIST2'] = HIST2

    # calculate HIST3
    FASTMA3 = talib.EMA(stock_data['close'].values, timeperiod=SHORT)
    SLOWMA3 = talib.EMA(stock_data['close'].values, timeperiod=MID_B)
    MACD3 = FASTMA3 - SLOWMA3
    SIGNAL3 = talib.EMA(MACD3, timeperiod=MID_B)
    HIST3 = MACD3 - SIGNAL3

    newdf['HIST3'] = HIST3

    # calculate HIST4
    FASTMA4 = talib.EMA(stock_data['close'].values, timeperiod=SHORT)
    SLOWMA4 = talib.EMA(stock_data['close'].values, timeperiod=LONG_B)
    MACD4 = FASTMA4 - SLOWMA4
    SIGNAL4 = talib.EMA(MACD4, timeperiod=LONG_B)
    HIST4 = MACD4 - SIGNAL4

    newdf['HIST4'] = HIST4

    # calculate HIST5
    FASTMA5 = talib.EMA(stock_data['close'].values, timeperiod=SHORT)
    SLOWMA5 = talib.EMA(stock_data['close'].values, timeperiod=MID_C)
    MACD5 = FASTMA5 - SLOWMA5
    SIGNAL5 = talib.EMA(MACD5, timeperiod=MID_C)
    HIST5 = MACD5 - SIGNAL5

    newdf['HIST5'] = HIST5

    # calculate MACD6
    FASTMA6 = talib.EMA(stock_data['close'].values, timeperiod=SHORT)
    SLOWMA6 = talib.EMA(stock_data['close'].values, timeperiod=LONG_C)
    MACD6 = FASTMA6 - SLOWMA6

    newdf['MACD6'] = MACD6

    return newdf


#
# talib_adx
#


def talib_adx(stock_data, LENGTH = 14):
    ''' Function to calculate ADX - Average Directional Movement Index

    Explain
    =======
    calculate ADX

    Input
    =====
    stock_data: DataFrame
        with OHLC, 'volume' and 'code' in each row

    Output
    ======
    Return: DataFrame
        with 'ADX' in each row, use the same index from Input stock_data
    '''

    # create a new empty output dataframe, sharing the same index
    newdf = pd.DataFrame(index=stock_data.index.copy())

    # calculate ADX
    newdf['ADX'] = talib.ADX(stock_data['high'].values,
                             stock_data['low'].values,
                             stock_data['close'].values, timeperiod=LENGTH)

    return newdf


#
# talib_atr
#


def talib_atr(stock_data, LENGTH = 14):
    '''Function to calculate ATR
    
    Input
    =====
    stock_data: DataFrame
        with OHLC, 'volume' and 'code' in each row
    
    Output
    ======
    Return: DataFrame
        with 'ADX' in each row, use the same index from Input stock_data
    '''
    # create a new empty output dataframe, sharing the same index
    newdf = pd.DataFrame(index=stock_data.index.copy())

    # calculate ATR
    newdf['ATR'] = talib.ATR(stock_data['high'].values,
                             stock_data['low'].values,
                             stock_data['close'].values, timeperiod=LENGTH)

    return newdf


#
# talib_nbarlow
#


def talib_nbarlow(stock_data, N_BAR_LOWEST = 10):
    ''' Funtion to calculate N-bar lowest
    
    Explain
    =======
    Calculate N'bar lowest
    
    Input
    =====
    stock_data: DataFrame
        with 'low' and 'code' in each row

    Output
    ======
    Return: DataFrame
        with 'LOW<N>' in each row, use the same index from Input stock_data
        Eg. 'LOW10'
    '''
    # create a new empty output dataframe, sharing the same index
    newdf = pd.DataFrame(index=stock_data.index.copy())
    
    # column name 'LOW<N>'
    nbar_lowest_col_name = 'LOW' + str(N_BAR_LOWEST)
    
    # calculate N days low
    newdf[nbar_lowest_col_name] = talib.MIN(stock_data['low'].values, timeperiod = N_BAR_LOWEST)
    newdf[nbar_lowest_col_name] = newdf[nbar_lowest_col_name].shift(1)

    return newdf


# define squeeze CONST
CONST_SQUEEZE_RELEASED = -1
CONST_SQUEEZE_ONGOING = 1

#
# is_squeeze_buy_point
#


def is_squeeze_buy_point(stock_data, index):
    ''' Function to tell whether the current index'ed bar a squeeze buy-point?
    
    RULE
    ====
    Return True when:
        a) TTM Wave C, ie. 'HIST5' and 'MACD6', must be greater than '0'. Then,
        b) 'SQUEEZE' should be either ongoing, or on the first bar of releasing.
        
    Input
    =====
    stock_data: DataFrame
        with features 'SQUEEZE', TTM Wave C/B/A, and ADX ready
    index: Index
        current index
    
    Output
    ======
    Return: boolean
        True or False
    '''
    ret = False
    
    #debug
    verbose = False

    # test TTM Wave C > 0
    if (np.isnan(stock_data.loc[index, 'HIST5']) or  # Handle NaN
        np.isnan(stock_data.loc[index, 'MACD6']) or
        (stock_data.loc[index, 'HIST5'] <= 0) or
        (stock_data.loc[index, 'MACD6'] <= 0)):
        
        if verbose:
            print(stock_data.loc[index, 'HIST5'], stock_data.loc[index, 'MACD6'])
            print(index, 'HIST5 or MACD6 <= 0, return False')
            
        ret = False
        return ret
    
    # test whether 'SQUEEZE' is on-going
    if (stock_data.loc[index, 'SQUEEZE'] == CONST_SQUEEZE_ONGOING):

        if verbose:
            print(index, ' Squeeze is on-going. return True')
        ret = True
        return ret
    
    # test whether on the first bar of 'SQUEEZE' release
    if (stock_data.loc[index, 'SQUEEZE'] == CONST_SQUEEZE_RELEASED):
        # check previous bar is 'SQUEEZE' ongoing?
        if (stock_data['SQUEEZE'].shift(1)[index] == CONST_SQUEEZE_ONGOING):
            
            if verbose:
                print(index, ' squeeze is released. But yesterday it is ongoing. return True')
            ret = True
            return ret
    
    if verbose:
        print(index, ' nothing hit. return False')

    return ret


CONST_SELL_REASON_STOP_LOSS = 1
CONST_SELL_REASON_N_LOW =2

#
# get_sell_point
#


def get_sell_point(stock_data, buy_index, multi_atr = 2., n_low = 10):
    ''' Function to get sell point
    
    Explain
    =======
    Find the sell point
    
    Input
    =====
    stock_data: DataFrame
        stock data with features
    buy_index: Index
        the bar where we start the trade
    multi_atr: float
        multiple of ATR as stop loss at
    n_low: int
        close breaks n_low bar's low
    
    Output
    ======
    sell_index: Index
        index of the sell point, or
        0, if cannot find sell point before the end
    sell_reason: int
        1 for stop loss, or 2 for N-bar low breakthrough.
        0 if not reaching any sell point.

    Features Required 
    =================
    'close', 'ATR'
    
    RULE
    ====
    Check each bar after buy_index, 
        a) close price is lower than (multi_atr * ATR)
        b) close lower than prvious n_low bars' low
    '''
    sell_index = 0
    sell_reason = 0
    
    # where we start the trade
    buy_price = stock_data['close'][buy_index]
    stop_price = buy_price - stock_data['ATR'][buy_index] * multi_atr
    #print('buy_price: ', buy_price)
    #print('stop_price: ', stop_price)
    
    buy_index_iloc = stock_data.index.get_loc(buy_index)
    remaining_stock_data = stock_data.iloc[(buy_index_iloc + 1):]
    n_low_col_name = 'LOW' + str(n_low)
            
    # check when to sell
    for index, row in remaining_stock_data.iterrows():
        if row['close'] <= stop_price: # down-break stop_price
            # print('today close is lowet than stop_price')
            # print(row['close'])
            # print(stop_price)
            sell_index = index
            sell_reason = CONST_SELL_REASON_STOP_LOSS
            break
        
        # close at a price lower than previous n_low bars' low
        if row['close'] <= row[n_low_col_name]:
            # print(row[['close', col_name]])
            sell_index = index
            sell_reason = CONST_SELL_REASON_N_LOW
            break
    
    return sell_index, sell_reason


