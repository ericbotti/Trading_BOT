import numpy as np
from BacktestRunner import Backtest_Traditional

class MovingAverageStrategy(Backtest_Traditional):
    def __init__(self, csv_path, date_col, max_holding):
        super().__init__(csv_path, date_col, max_holding)

    def generate_signals(self):
        df = self.dmgt.df
        df['ma_20'] = df.close.rolling(20).mean()
        df['ma_50'] = df.close.rolling(50).mean()
        df['ma_diff'] = df.ma_20 - df.ma_50
        df['longs'] = ((df.ma_diff > 0) & (df.ma_diff.shift(1) < 0)) * 1
        df['shorts'] = ((df.ma_diff < 0) & (df.ma_diff.shift(1) > 0)) * -1
        df['entry'] = df.longs + df.shorts
        self.dmgt.df = df

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

class IMeanReversion(Backtest_Traditional):
    def __init__(self, csv_path, date_col, max_holding, ub_mult, lb_mult, up_filter, down_filter, long_lookback, short_lookback):
        super().__init__(csv_path, date_col, max_holding)
        
        # Target and stop losses mults
        self.ub_mult = ub_mult
        self.lb_mult = lb_mult
        # Conditions
        self.up_filter = up_filter
        self.down_filter = down_filter
        # Lookbacks
        self.long_lookback = long_lookback # n
        self.short_lookback = short_lookback # k

    def generate_signals(self):
        df = self.dmgt.df
        df['min_24'] = df.close.rolling(self.short_lookback).min()
        df['max_24'] = df.close.rolling(self.short_lookback).max()
        df['longs'] = ((df.close <= df.min_24) & (df.close > df.close.shift(self.long_lookback)*self.up_filter))*1
        df['shorts'] = ((df.close >= df.max_24) & (df.close < df.close.shift(self.long_lookback) * self.down_filter)) * -1
        df['entry'] = df.longs + df.shorts
        df.dropna(inplace=True)
        df = self.dmgt.df

class PolyTrend(Backtest_Traditional):
    def __init__(self, csv_path, date_col, max_holding, lookback, look_ahead, long_thres, short_thres, ub_mult, lb_mult, long_tp, long_sl, short_tp, short_sl):
        super().__init__(csv_path, date_col, max_holding)

        self.lookback = lookback
        self.look_ahead = look_ahead
        self.long_thres = long_thres
        self.short_thres = short_thres
        self.ub_mult = ub_mult
        self.lb_mult = lb_mult
        self.long_tp = long_tp
        self.long_sl = long_sl
        self.short_tp = short_tp
        self.short_sl = short_sl

    # -- Functions that performs the polynomial rolling time trend. It receives the close price of BPF in pandas series format and n, the lookahead from constructor -- #
    @staticmethod
    def rolling_tt(series, n):
        y = series.values.reshape(-1, 1)
        t = np.arange(len(y))
        X = np.c_[np.ones_like(y), t, t ** 2]
        betas = np.linalg.inv(X.T @ X) @ X.T @ y
        new_vals = np.array([1, t[-1]+n, (t[-1]+n)**2])
        pred = new_vals@betas  # beta0 + beta1 * t[-1]+n + beta2 * (t[-1]+n)**2
        return pred

    def generate_signals(self):
        df = self.dmgt.df
        n = self.look_ahead
        df['preds'] = df.close.rolling(self.lookback).apply(self.rolling_tt, args=(n,), raw=False)
        # Predicted change column
        df['pdelta'] = (df.preds/df.close)-1
        df['longs'] = (df.pdelta > self.long_thres)*1
        df['shorts'] = (df.pdelta < self.short_thres)*-1
        df['entry'] = df.longs + df.shorts
        df.dropna(inplace=True)
        df = self.dmgt.df

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

    # IntradayMeanReversion parameters
    up_filter = 1.03 # change this to change filter for longs
    down_filter = 0.97 # change this for shorts filter
    long_lookback = 60*24*30
    short_lookback = 60*24

    # PolyTimeTrend parameters
    lookback = 12*24 # I fixed this at 24 hour minimum, 60min = 24, 30min = 48 etc
    look_ahead = 4 # periods ahead to predict for each endpoint of the model
    long_thres = 0.03 #if model predicts a price that represents an increase of 3% we enter
    short_thres = -0.03 # opposite of above
    long_tp = 1.03 #long target in %
    long_sl = 0.98 # long stop loss 
    short_tp = 0.96 # short target 
    short_sl = 1.025 # short stop loss


    # Comment/uncomment the test you want to start
    #system = MovingAverageStrategy(csv_path, date_col, maximum_holding)
    #system = MomentumRSI(csv_path, date_col, maximum_holding, ub_mult, lb_mult, rsi_window, rsi_long, rsi_short, ma_long, ma_short)
    system = HigherLower(csv_path, date_col, maximum_holding)
    #system = IMeanReversion(csv_path, date_col, maximum_holding, ub_mult, lb_mult, up_filter, down_filter, long_lookback, short_lookback)
    #system = PolyTrend(csv_path, date_col, maximum_holding, lookback, look_ahead, long_thres, short_thres, ub_mult, lb_mult, long_tp, long_sl, short_tp, short_sl)
    
    # Change time frequency
    system.dmgt.change_resolution('240min') # 1min .... 60 min

    system.run_backtest()
    system.show_performace()
    system.save_backtest()