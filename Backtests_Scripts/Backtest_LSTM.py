import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BacktestRunner import Backtest_LSTM
from TechnicalIndicator import get_technical_indicators
import tensorflow as tf
from tensorflow import keras
from pickle import load

# ---- Contains the LSTM algorithm ---- #
class LSTM(Backtest_LSTM):

    def __init__(self, csv_path, max_holding):
        super().__init__(csv_path, max_holding)

    # -- Function that validates the data -- #
    def data_validation(self, data):
        df = data.copy()
        # Change timestamp column to datetime 
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Drop duplicates
        df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
        df.reset_index(inplace=True)
        keep_cols = ['volume','open','low','high','close','timestamp','t_plus']
        for i in df.columns:
            if i in keep_cols:
                pass
            else:
                del df[i]
        # Add technical indicators to the dataset
        df = get_technical_indicators(df)
        # Reorder columns
        columns_titles = ['timestamp','open','low','high','close','volume','t_plus','ma5','ma20','ma30','ma60','26ema','12ema','MACD','20sd','upper_band','lower_band','ema','RSI','log_ret_1','log_ret_2','log_ret_3','log_ret_4','log_ret_5','cum_log_ret_3','cum_log_ret_5','diff_cum_log_ret','ROC_9','ROC_14','DEMA_short','DEMA_long','momentum','log_momentum']
        df=df.reindex(columns=columns_titles)
        df = df.dropna()
        return df

    # -- Scale data that are going to be used in the LSTM algorithm -- #
    def scaler(self, df):
        # Selecting the Features
        features = ['low','high','open','volume','ma5','ma20','ma30','ma60','26ema','12ema','MACD','20sd','upper_band','lower_band','ema','RSI','log_ret_1','log_ret_2','log_ret_3','log_ret_4','log_ret_5','cum_log_ret_3','cum_log_ret_5','diff_cum_log_ret','ROC_9','ROC_14','DEMA_short','DEMA_long','momentum','log_momentum']
        # Load the scaler used during the training of the neural network
        scaler = load(open('../LSTM_MinMaxModels/MinMaxModel_test_69.pkl', 'rb'))
        # Add the scaler to the dataframe
        feature_transform = scaler.transform(df[features])
        feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=df.index)
        df = feature_transform
        df = np.array(df)
        df = df.reshape(df.shape[0], 1, df.shape[1])
        return df

    # -- Function that generate signals -- #
    def generate_signals(self):
        # Generate two dataframes with the correct format and columns. Data_processed are ready to be inserted in the LSTM algorithm
        df = self.dmgt.df
        data = self.data_validation(df)
        data_processed = self.scaler(data)
        previously = data.close.values.reshape(-1, 1)
        # LSTM 
        lstm = keras.models.load_model('../LSTM_Models/test_69.h5')
        pred = lstm(data_processed)
        res = np.array(tf.math.divide(pred,previously)-1)
        # Signal generator
        entry_cond = 0.05 # Difference in price and precition that has to present in order to generate a signal (not in %)
        data['long_entry'] = 1 * (res > entry_cond)
        data['short_entry'] = -1 * (res < -entry_cond)
        data['entry'] = data.long_entry + data.short_entry # Signal added to the dataframe
        self.dmgt.df = data

if __name__ == '__main__':
    # Universal parameters
    csv_path = '../Data/BPF_testset_1min.csv'
    maximum_holding = 60 # Minutes

    system = LSTM(csv_path, maximum_holding)

    system.run_backtest()
    system.show_performace()

    system.dmgt.df.to_csv('../Backtests_Data/Backtest_LSTM.csv')