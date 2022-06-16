import numpy as np
from BacktestRunner import Backtest_Traditional

class MomentumRSI(Backtest_Traditional):
    def __init__(self, csv_path, date_col, max_holding, ub_mult, lb_mult, rsi_window, rsi_long, rsi_short, ma_long, ma_short):
        super().__init__(csv_path, date_col, max_holding)

        self.ub_mult = ub_mult
        self.lb_mult = lb_mult
        # Rsi parameters
        self.rsi_window = rsi_window
        self.rsi_long = rsi_long
        self.rsi_short = rsi_short
        # Moving average parameters
        self.ma_long = ma_long
        self.ma_short = ma_short

    def calculate_rsi(self):
        df = self.dmgt.df
        # Create change column
        df['change'] = df.close.diff()
        df['U'] = [x if x > 0 else 0 for x in df.change] # Calculate column of upwards changes
        df['D'] = [abs(x) if x < 0 else 0 for x in df.change] # Create column of downwards changes
        df['U'] = df.U.ewm(span=self.rsi_window, min_periods=self.rsi_window-1).mean()
        df['D'] = df.D.ewm(span=self.rsi_window, min_periods=self.rsi_window - 1).mean()
        df['RS'] = df.U / df.D
        df['RSI'] = 100 - 100/(1+df.RS)
        df.drop(['change', 'U', 'D', 'RS'], axis=1, inplace=True)

    # -- Calculates two expontential movings averages, based on arguments passed into construtor -- #
    def calculate_ma(self):
        df = self.dmgt.df
        df['ma_long'] = df.close.ewm(span=self.ma_long, min_periods=self.ma_long-1).mean()
        df['ma_short'] = df.close.ewm(span=self.ma_short, min_periods=self.ma_short - 1).mean()

    def generate_signals(self):
        df = self.dmgt.df
        self.calculate_ma()
        self.calculate_rsi()
        df.dropna(inplace=True)
        # 1 if rsi < 30 & ma_short > ma_long, 0 otherwise
        df['longs'] = ((df.RSI < self.rsi_long) & (df.ma_short > df.ma_long))*1
        # -1 if rsi > 70 & ma_short < ma_long, 0 otherwise
        df['shorts'] = ((df.RSI > self.rsi_short) & (df.ma_short < df.ma_long))*-1
        df['entry'] = df.longs + df.shorts
        df.dropna(inplace=True)
        df = self.dmgt.df

class HigherLower(Backtest_Traditional):
    def __init__(self, csv_path, date_col, max_holding):
        super().__init__(csv_path, date_col, max_holding)

    def generate_signals(self):
        df = self.dmgt.df
        df['longs'] = ((df.high > df.high.shift(1)) & (df.high.shift(1) > df.high.shift(2)) & (df.close.shift(2) > df.high.shift(3))) * 1
        df['shorts'] = ((df.low < df.low.shift(1)) & (df.low.shift(1) < df.low.shift(2)) & (df.close.shift(2) < df.low.shift(3))) * -1
        df['entry'] = df.shorts + df.longs
        df.dropna(inplace=True)

if __name__ == '__main__':
    # Universal parameters
    csv_path = '../Data/BPF_testset_1min.csv'
    date_col = 'timestamp'
    maximum_holding = 55
    ub_mult = 1.03 # change this to change target (longs) stops (shorts)
    lb_mult = 0.97 # change this to change stops (longs) targets (shorts)

    # MomentumRSI parameters
    rsi_window = 14
    rsi_long = 30
    rsi_short = 70
    ma_long = 26
    ma_short = 12

    # Comment/uncomment the test you want to start
    #system = MomentumRSI(csv_path, date_col, maximum_holding, ub_mult, lb_mult, rsi_window, rsi_long, rsi_short, ma_long, ma_short)
    system = HigherLower(csv_path, date_col, maximum_holding)
    
    # Change time frequency
    system.dmgt.change_resolution('120min') # 1min .... 60 min

    system.run_backtest()
    system.show_performace()
    system.save_backtest()