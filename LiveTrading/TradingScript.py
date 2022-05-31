from Processor import Processor
from TechnicalIndicator import get_technical_indicators
from DeribitWS import DeribitWS
import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
from datetime import datetime 
import time
import json
from termcolor import colored
from pickle import load

with open('./auth_creds.json') as j:
    creds = json.load(j)

client_id = creds['paper']['client_id']
client_secret = creds['paper']['client_secret']


class TradingScript(Processor):
    def __init__(self, client_id, client_secret, instrument, timeframe, trade_capital, max_holding, ub_mult, lb_mult, entry_cond, lookback, n, live=False):
        super().__init__(client_id, client_secret, instrument, timeframe, trade_capital, max_holding, ub_mult, lb_mult, live)

        self.entry_cond = entry_cond
        self.lookback = lookback
        self.n = n
        self.delta = 60_000 # Milliseconds -> 60'000 equals to 1 min
        self.WS = DeribitWS(client_id, client_secret, live)

    def data_validation(self, data):
        df = data.copy()
        # Change timestamp column to datetime 
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Drop duplicates
        df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
        df.reset_index(inplace=True)
        keep_cols = ['volume', 'open', 'low', 'high', 'close', 'timestamp']
        for i in df.columns:
            if i in keep_cols:
                pass
            else:
                del df[i]
        df = get_technical_indicators(df)
        # Reorder columns
        columns_titles = ['timestamp','open','low','high','close','volume','ma5','ma20','ma30','ma60','26ema','12ema','MACD','20sd','upper_band','lower_band','ema','RSI','log_ret_1','log_ret_2','log_ret_3','log_ret_4','log_ret_5','cum_log_ret_3','cum_log_ret_5','diff_cum_log_ret','ROC_9','ROC_14','DEMA_short','DEMA_long','momentum','log_momentum']
        df=df.reindex(columns=columns_titles)
        # Selecting features
        features = ['low','high','open','volume','ma5','ma20','ma30','ma60','26ema','12ema','MACD','20sd','upper_band','lower_band','ema','RSI','log_ret_1','log_ret_2','log_ret_3','log_ret_4','log_ret_5','cum_log_ret_3','cum_log_ret_5','diff_cum_log_ret','ROC_9','ROC_14','DEMA_short','DEMA_long','momentum','log_momentum']
        df = df.dropna()
        # Load the scaler
        scaler = load(open('../LSTM_MinMaxModels/MinMaxModel_test_69.pkl', 'rb'))
        # Add the scaler to the dataframe
        feature_transform = scaler.transform(df[features])
        feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=df.index)
        df = feature_transform
        df = np.array(df)
        df = df.reshape(df.shape[0], 1, df.shape[1])
        return df

    def generate_signal(self, data,lstm):
        previously = data.close.values.reshape(-1, 1)
        y = self.data_validation(data)
        pred = lstm(y)
        now = datetime.now()
        dt_string = now.strftime('%d/%m/%Y %H:%M:%S') # Generate a readable time format
        print(colored(f'{dt_string}', 'cyan', attrs=['bold']), f"Predicted: {int(pred)}, Last: {int(previously[-1])}")
        if (pred / previously[-1] - 1) > self.entry_cond:
            return 1
        elif (pred / previously[-1] - 1) < -self.entry_cond:
            return -1
        else:
            return 0

    def get_data(self):
        end = self.utc_times_now()
        start = end - self.delta*self.lookback
        json_resp = self.WS.get_data(self.instrument, start, end, self.timeframe)
        if 'error' in json_resp.keys():
            print(colored(json_resp['error'],'red'))
            return False, False
        elif 'result' in json_resp.keys():
            data = self.json_to_dataframe(json_resp)
            return True, data
        else:
            return False, False # False, do we have data -> false


    def run(self, endtime):
        print('\n\n')
        print(colored('=======================================','red'))
        print(colored('=======================================','red'))
        print(colored('==                                   ==','red'))
        print(colored('== Eric Bottinelli - Bachelor Thesis ==','red'))
        print(colored('==            Trading Bot            ==','red'))
        print(colored('==                                   ==','red'))
        print(colored('=======================================','red'))
        print(colored('=======================================','red'))
        print('\n\n')
        now = datetime.now()
        dt_string = now.strftime('%d/%m/%Y %H:%M:%S') # Generate a readable time format
        initial_equity = self.WS.account_summary('BTC')["result"]["equity"]
        print(colored(f"Started strategy at {dt_string} with {initial_equity} BTC equity",'magenta'), '\n')
        while True:
            timenow = datetime.now()
            if timenow.second == 0:
                t = time.time()
                good_call, data = self.get_data()
                if not good_call:
                    time.sleep(1)
                    continue
                last_price = data.close.values[-1]
                lstm = keras.models.load_model('../LSTM_Models/test_69.h5')
                signal = self.generate_signal(data,lstm)

                if signal == 1 and self.open_pos is False:
                    self.open_long()
                    print(colored(f"Took {time.time()-t} seconds to execute",'yellow'))
                elif signal == -1 and self.open_pos is False:
                    self.open_short()
                    print(colored(f"Took {time.time() - t} seconds to execute", 'yellow'))
                elif self.open_pos:
                    self.monitor_open(last_price, initial_equity)
                else:
                    pass
                time.sleep(1)

            if timenow >= endtime:
                print(colored(f"Exiting strategy at {dt_string}", 'blue'))
                if self.open_pos:
                    self.close_position(initial_equity)
                break

if __name__ == '__main__':
    instrument = 'BTC-PERPETUAL'
    timeframe = '1'
    trade_capital = 90
    ub_mult = 1.04
    lb_mult = 0.96
    max_holding = 55 # Minutes
    entry_cond = 0.05
    n = 3
    lookback = 59

    strat = TradingScript(client_id, client_secret, instrument, timeframe, trade_capital, max_holding, ub_mult, lb_mult, entry_cond, lookback, n, live=False) # Creation of the object

    endtime = datetime(2022, 6, 19, 9, 45)
    strat.run(endtime)