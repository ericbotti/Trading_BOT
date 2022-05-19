from DeribitWS import DeribitWS
import pandas as pd
import numpy as np
import datetime as dt
import json
import time

from termcolor import colored


class StrategyBase:

    def __init__(self, client_id, client_secret, instrument, timeframe,
                 trade_capital, max_holding, ub_mult, lb_mult, live=False):

        self.WS = DeribitWS(client_id, client_secret, live)
        self.instrument = instrument
        self.timeframe = timeframe
        self.trade_capital = trade_capital

        self.ub_mult = ub_mult
        self.lb_mult = lb_mult
        self.max_holding = max_holding
        self.max_holding_limit = max_holding

        #trade variables
        self.open_pos = False
        self.stop_price = None
        self.target_price = None
        self.direction = None
        self.fees = 0
        self.open_price = None
        self.close_price = None

        self.trades = {'open_timestamp': [], 'close_timestamp': [],
                       'open': [], 'close': [], 'fees': [], 'direction': []}


    @staticmethod
    def json_to_dataframe(json_resp):
        res = json_resp['result']
        df = pd.DataFrame(res)
        df['ticks_'] = df.ticks / 1000
        df['timestamp'] = [dt.datetime.utcfromtimestamp(date) for date in df.ticks_]
        return df

    @staticmethod
    def utc_times_now():
        string_time = time.strftime("%Y %m %d %H %M %S").split(' ')
        int_time = list(map(int, string_time))
        now = dt.datetime(int_time[0],
                          int_time[1],
                          int_time[2],
                          int_time[3],
                          int_time[4],
                          int_time[5]).timestamp() * 1000
        return now 

    def open_long(self):
        trade_resp = self.WS.market_order(self.instrument, self.trade_capital, 'long')
        prev_equity = self.WS.account_summary('BTC')["result"]["equity"]
        print('\n')
        if 'result' in trade_resp.keys():
            self.open_pos = True
            self.open_price = trade_resp['result']['order']['average_price']
            self.target_price = self.open_price * self.ub_mult
            self.stop_price = self.open_price * self.lb_mult
            self.direction = 1
            self.fees += trade_resp['result']['order']['commission']
            self.trades['open_timestamp'].append(dt.datetime.now())
            now = dt.datetime.now()
            dt_string = now.strftime('%d/%m/%Y %H:%M:%S') # Generate a readable time format
            print(colored(f'{dt_string}', 'cyan', attrs=['bold']), colored(f'Opening', attrs=['bold']), colored('long', 'green', attrs=['bold']), colored(f'at {self.open_price} with stop price {round(self.stop_price,2)} and target {round(self.target_price,2)}', attrs=['bold']))

        else:
            print(colored(trade_resp['error'], 'red'))


    def open_short(self):
        trade_resp = self.WS.market_order(self.instrument, self.trade_capital, 'short')
        print('\n')
        if 'result' in trade_resp.keys():
            self.open_pos = True
            self.open_price = trade_resp['result']['order']['average_price']
            self.target_price = self.open_price * self.lb_mult
            self.stop_price = self.open_price * self.ub_mult
            self.direction = -1
            self.fees += trade_resp['result']['order']['commission']
            self.trades['open_timestamp'].append(dt.datetime.now())
            now = dt.datetime.now()
            dt_string = now.strftime('%d/%m/%Y %H:%M:%S') # Generate a readable time format
            print(colored(f'{dt_string}', 'cyan', attrs=['bold']), colored(f'Opening', attrs=['bold']), colored('short', 'red', attrs=['bold']), colored(f'at {self.open_price} with stop price {round(self.stop_price,2)} and target {round(self.target_price,2)}', attrs=['bold']))
        else:
            print(colored(trade_resp['error'], 'red'))


    def reset_vars(self):
        self.open_pos = False
        self.target_price = None
        self.stop_price = None
        self.direction = None
        self.fees = 0
        self.open_price = None
        self.max_holding = self.max_holding_limit
        self.close_price = None

    def close_position(self,initial_equity):
        now = dt.datetime.now()
        dt_string = now.strftime('%d/%m/%Y %H:%M:%S') # Generate a readable time format
        if self.direction == 1:
            close_resp = self.WS.market_order(self.instrument, self.trade_capital, 'short')
            if 'result' in close_resp.keys():
                self.close_price = close_resp['result']['order']['average_price']
                self.trades["open"].append(self.open_price)
                self.trades["close"].append(self.close_price)
                self.trades["direction"].append(self.direction)
                self.trades["fees"].append(self.fees + close_resp['result']['order']['commission'])
                self.trades['close_timestamp'].append(dt.datetime.now())
                if round(self.close_price / self.open_price - 1.001, 5) >= 0:
                    print(colored(f'{dt_string}', 'cyan', attrs=['bold']), colored(f'Closing long at {self.close_price} for a', attrs=['bold']), colored(f'{round(self.close_price / self.open_price - 1.001, 5) * 100}% return', 'green', attrs=['bold']))
                else:
                    print(colored(f'{dt_string}', 'cyan', attrs=['bold']), colored(f'Closing long at {self.close_price} for a', attrs=['bold']), colored(f'{round(self.close_price / self.open_price - 1.001, 5) * 100}% return', 'red', attrs=['bold']))
                self.reset_vars()
            else:
                print(colored(close_resp['error'], 'red'))

        if self.direction == -1:
            close_resp = self.WS.market_order(self.instrument, self.trade_capital, 'long')
            if 'result' in close_resp.keys():
                self.close_price = close_resp['result']['order']['average_price']
                self.trades["open"].append(self.open_price)
                self.trades["close"].append(self.close_price)
                self.trades["direction"].append(self.direction)
                self.trades["fees"].append(self.fees + close_resp['result']['order']['commission'])
                self.trades['close_timestamp'].append(dt.datetime.now())
                if (-1* (self.close_price/self.open_price - 1)) >= 0:
                    print(colored(f'{dt_string}', 'cyan', attrs=['bold']), colored(f'Closing short at {self.close_price} for a', attrs=['bold']), colored(f'{round(self.close_price/self.open_price - 1.001, 5) * -1 * 100}% return','green', attrs=['bold']))
                else:
                    print(colored(f'{dt_string}', 'cyan', attrs=['bold']), colored(f'Closing short at {self.close_price} for a', attrs=['bold']), colored(f'{round(self.close_price/self.open_price - 1.001, 5) * -1 * 100}% return','red', attrs=['bold']))
                self.reset_vars()
            else:
                print(colored(close_resp['error'], 'red'))

        current_equity = self.WS.account_summary('BTC')["result"]["equity"]
        gain = current_equity - initial_equity
        '''
        end = self.utc_times_now()
        start = end - 1
        instrument = 'BTC-PERPETUAL'
        timeframe = '1'
        btc_price = self.WS.get_data(instrument, start, end, timeframe)
        btc_price = int(btc_price[0])
        #btc_price = self.json_to_dataframe(btc_price)
        #btc_price = btc_price.astype(int)
'''
        profit_loss = self.WS.get_order_history_by_instrument("BTC-PERPETUAL")
        print(profit_loss)
        #print(f'Gain of this trading session: {gain} BTC equivalent to {gain * float(self.close_price)} USD')
        print(f'Current equity: {current_equity}')

    def monitor_open(self, price, initial_equity):
        if price >= self.target_price and self.direction == 1:
            self.close_position(initial_equity)
            print(colored('Long target hit', 'blue'))
        elif price <= self.stop_price and self.direction == 1:
            self.close_position(initial_equity)
            print(colored('Long stop hit', 'blue'))
        elif price <= self.target_price and self.direction == -1:
            self.close_position(initial_equity)
            print(colored('Short target hit', 'blue'))
        elif price >= self.stop_price and self.direction == -1:
            self.close_position(initial_equity)
            print(colored('Short stop hit', 'blue'))
        elif self.max_holding <= 0:
            self.close_position(initial_equity)
            print(colored("Max holding time exceeded closing position", 'blue'))
        else:
            self.max_holding = self.max_holding - 1