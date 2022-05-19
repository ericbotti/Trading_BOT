import pandas as pd
import numpy as np

def compute_RSI (data, time_window):
    diff = data.diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

def compute_WilliamsR(high, low, close, lookback):
    highh = high.rolling(lookback).max() 
    lowl = low.rolling(lookback).min()
    wr = -100 * ((highh - close) / (highh - lowl))
    return wr

def compute_Log_Return(data, lag):
    df = data.copy()
    df['log_return'] = np.log1p(df.open.pct_change(lag))
    return df['log_return']

def DEMA(data, time_period):
    #Calculate the Exponential Moving Average for some time_period (in days)
    EMA = data['open'].ewm(span=time_period, adjust=False).mean()
    #Calculate the DEMA
    DEMA = 2*EMA - EMA.ewm(span=time_period, adjust=False).mean()
    return DEMA

def get_technical_indicators(dataset):
    # Create 5, 20, 30 and 60 min Moving Average from Paper 21
    dataset['ma5'] = dataset['open'].rolling(window=5).mean()
    dataset['ma20'] = dataset['open'].rolling(window=20).mean()
    dataset['ma30'] = dataset['open'].rolling(window=30).mean()
    dataset['ma60'] = dataset['open'].rolling(window=60).mean()
    
    # Create MACD
    dataset['26ema'] = dataset['open'].ewm(span=26).mean()
    dataset['12ema'] = dataset['open'].ewm(span=12).mean()
    dataset['MACD'] = dataset['12ema']-dataset['26ema']

    # Create Bollinger Bands
    dataset['20sd'] = dataset['open'].rolling(window = 21).std()
    dataset['upper_band'] = dataset['ma20'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma20'] - (dataset['20sd']*2)
    
    # Create Exponential moving average
    dataset['ema'] = dataset['open'].ewm(com=0.9).mean()

    # Create RSI
    dataset['RSI'] = compute_RSI(dataset['open'], 14)

    # Create Williams' %R
    dataset['wr_14'] = compute_WilliamsR(dataset['high'], dataset['low'], dataset['close'], 14)
    
    # Create Log-Returns
    dataset['log_ret_1'] = compute_Log_Return(dataset, 1)
    dataset['log_ret_2'] = compute_Log_Return(dataset, 2)
    dataset['log_ret_3'] = compute_Log_Return(dataset, 3)
    dataset['log_ret_4'] = compute_Log_Return(dataset, 4)
    dataset['log_ret_5'] = compute_Log_Return(dataset, 5)

    # Create cumulative sum of last 3 and 5 minutes of log-returns
    data = dataset.copy()
    data.iloc[:, 20:23] = dataset.iloc[:, 20:23].cumsum(axis=1, skipna=False)
    data.iloc[:, 20:25] = dataset.iloc[:, 20:25].cumsum(axis=1, skipna=False)
    dataset['cum_log_ret_3'] = data['log_ret_3']
    dataset['cum_log_ret_5'] = data['log_ret_5']

    # Create difference between cumulative sum of last 3 and 5 minutes of log-returns
    dataset['diff_cum_log_ret'] = (dataset['cum_log_ret_5'] - dataset['cum_log_ret_3'])

    # Create Rate of Change (ROC) for 9 and 14 minutes period
    dataset['ROC_9'] = dataset['open'].diff(9)
    dataset['ROC_14'] = dataset['open'].diff(14)

    # Create Double Exponential Moving Average (DEMA)
    dataset['DEMA_short'] = DEMA(dataset, 20)
    dataset['DEMA_long'] = DEMA(dataset, 50)
    
    # Create Momentum
    dataset['momentum'] = dataset['open']-5
    dataset['log_momentum'] = np.log(dataset['momentum'])
    return dataset
