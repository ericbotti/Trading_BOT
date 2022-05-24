from Datamanager import DataManager_LSTM, DataManager_Traditional
import matplotlib.pyplot as plt
import pandas as pd

# ---- Class that does the LSTM strategy backtest ---- #
class Backtest_LSTM:
    def __init__(self, csv_path, maximum_holding):
        self.dmgt = DataManager_LSTM(csv_path)
        # Trade variables
        self.open_pos = False
        self.entry_price = None
        self.direction = None
        self.target_price = None
        self.stop_price = None
        # Vertical barrier variable
        self.max_holding = maximum_holding
        self.max_holding_limit = maximum_holding
        # Parameters
        self.ub_mult = 1.03
        self.lb_mult = 0.97
        #self.lb_mult = self.dmgt.df.close.ewm(span=50).std()
        # Special case of vertical barrier
        self.end_date = self.dmgt.df.index.values[-1]

        self.returns_series = []
        self.holding_series = []
        self.direction_series = []

    # -- Function that receive the price at which the long position should bee initiated and populates trade variables from constructor with relevant variables -- #
    def open_long(self, price):
        self.open_pos = True
        self.direction = 1
        self.entry_price = price
        self.target_price = price * self.ub_mult
        self.stop_price = price * self.lb_mult
        self.add_zeros()

    # -- Function that receive the price at which the short position should bee initiated and populates trade variables from constructor with relevant variables -- #
    def open_short(self, price):
        self.open_pos = True
        self.direction = -1
        self.entry_price = price
        self.target_price = price * self.lb_mult
        self.stop_price = price * self.ub_mult
        self.add_zeros()

    # -- Resets the variables after we close a trade -- #
    def reset_variables(self):
        self.open_pos = False
        self.entry_price = None
        self.direction = None
        self.target_price = None
        self.stop_price = None
        self.max_holding = self.max_holding_limit

    # -- Function that appends zeros to the missing slots -- #
    def add_zeros(self):
        self.returns_series.append(0)
        self.holding_series.append(0)
        self.direction_series.append(0)

    # -- Receives the exite price and appends the trade Profit & Loss (pnl) to the returns series and resets variables -- #
    def close_position(self, price):
        pnl = (price / self.entry_price - 1) * self.direction
        self.process_close_var(pnl)
        self.reset_variables()

    # -- Update parameters -- #
    def process_close_var(self, pnl):
        self.returns_series.append(pnl)
        self.direction_series.append(self.direction)
        holding = self.max_holding_limit - self.max_holding
        self.holding_series.append(holding)

    # -- Function that makes sure generated signal has been included in the child class -- #
    def generate_signals(self):
        if 'entry' not in self.dmgt.df.columns:
            raise Exception('You have not created signals yet')

    def monitor_open_positions(self, price, timestamp):
        # Check upper horizontal barrier for long positions
        if price >= self.target_price and self.direction == 1:
            self.close_position(price)
        # Check lower horizontal barrier for long positions
        elif price <= self.stop_price and self.direction == 1:
            self.close_position(price)
        # Check lower horizontal barrier for short positions
        elif price <= self.target_price and self.direction == -1:
            self.close_position(price)
        # Check upper horizontal barrier for short positions
        elif price >= self.stop_price and self.direction == -1:
            self.close_position(price)
        # Check special case of vertical barrier
        elif timestamp == self.end_date:
            self.close_position(price)
        # Check vertical barrier
        elif self.max_holding <= 0:
            self.close_position(price)
        # If all above conditions are not true, decrement max holding by 1 and append a zero to returns column
        else:
            self.max_holding = self.max_holding - 1
            self.add_zeros()

    # -- Functions that merges the new columns created for the backtest into the dataframe, also resets the return series to empty list -- #
    def add_trade_cols(self):
        self.dmgt.df['returns'] = self.returns_series
        self.dmgt.df['holding'] = self.holding_series
        self.dmgt.df['direction'] = self.direction_series

        self.returns_series = []
        self.holding_series = []
        self.direction_series = []

    # -- Backtest heart -- #
    def run_backtest(self):
        # Signals generated from child class
        self.generate_signals()
        # Loop over dataframe
        for row in self.dmgt.df.itertuples():
            # If get a long signal and do not have open position -> open a long position
            if row.entry == 1 and self.open_pos is False:
                self.open_long(row.t_plus)
            # If get a short position and do not have open position -> open a short position
            elif row.entry == -1 and self.open_pos is False:
                self.open_short(row.t_plus)
            # Monitor open positions to see if any of the barriers have been touched, see function above
            elif self.open_pos:
                self.monitor_open_positions(row.close, row.Index)
            else:
                self.add_zeros()
        self.add_trade_cols()
    
    # -- Show a performance graph of the backtest -- #
    def show_performace(self):
        self.dmgt.df.returns.cumsum().plot()
        plt.title(f"Strategy results for {self.dmgt.timeframe} timeframe")
        plt.show()

    def save_backtest(self):
        strat_name = self.__class__.__name__
        tf = self.dmgt.timeframe
        self.dmgt.df.to_csv(f"../Backtests_Data/{strat_name}_{tf}.csv")

# ---- Class that does backtest for traditional strategies ---- #
class Backtest_Traditional:
    def __init__(self, csv_path, date_col, maximum_holding):
        self.dmgt = DataManager_Traditional(csv_path, date_col)
        # Trade variables
        self.open_pos = False
        self.entry_price = None
        self.direction = None
        self.target_price = None
        self.stop_price = None
        # Vertical barrier variable
        self.max_holding = maximum_holding
        self.max_holding_limit = maximum_holding
        # Parameters
        self.ub_mult = 1.03
        self.lb_mult = 0.97
        # Special case of vertical barrier
        self.end_date = self.dmgt.df.index.values[-1]

        self.returns_series = []
        self.holding_series = []
        self.direction_series = []

    # -- Function that receive the price at which the long position should bee initiated and populates trade variables from constructor with relevant variables -- #
    def open_long(self, price):
        self.open_pos = True
        self.direction = 1
        self.entry_price = price
        self.target_price = price * self.ub_mult
        self.stop_price = price * self.lb_mult
        self.add_zeros()

    # -- Function that receive the price at which the short position should bee initiated and populates trade variables from constructor with relevant variables -- #
    def open_short(self, price):
        self.open_pos = True
        self.direction = -1
        self.entry_price = price
        self.target_price = price * self.lb_mult
        self.stop_price = price * self.ub_mult
        self.add_zeros()

    # -- Resets the variables after we close a trade -- #
    def reset_variables(self):
        self.open_pos = False
        self.entry_price = None
        self.direction = None
        self.target_price = None
        self.stop_price = None
        self.max_holding = self.max_holding_limit

    # -- Function that appends zeros to the missing slots -- #
    def add_zeros(self):
        self.returns_series.append(0)
        self.holding_series.append(0)
        self.direction_series.append(0)

    # -- Receives the exite price and appends the trade Profit & Loss (pnl) to the returns series and resets variables -- #
    def close_position(self, price):
        pnl = (price / self.entry_price - 1) * self.direction
        self.process_close_var(pnl)
        self.reset_variables()

    # -- Update parameters -- #
    def process_close_var(self, pnl):
        self.returns_series.append(pnl)
        self.direction_series.append(self.direction)
        holding = self.max_holding_limit - self.max_holding
        self.holding_series.append(holding)

    # -- Function that makes sure generated signal has been included in the child class -- #
    def generate_signals(self):
        if 'entry' not in self.dmgt.df.columns:
            raise Exception('You have not created signals yet')

    def monitor_open_positions(self, price, timestamp):
        # Check upper horizontal barrier for long positions
        if price >= self.target_price and self.direction == 1:
            self.close_position(price)
        # Check lower horizontal barrier for long positions
        elif price <= self.stop_price and self.direction == 1:
            self.close_position(price)
        # Check lower horizontal barrier for short positions
        elif price <= self.target_price and self.direction == -1:
            self.close_position(price)
        # Check upper horizontal barrier for short positions
        elif price >= self.stop_price and self.direction == -1:
            self.close_position(price)
        # Check special case of vertical barrier
        elif timestamp == self.end_date:
            self.close_position(price)
        # Check vertical barrier
        elif self.max_holding <= 0:
            self.close_position(price)
        # If all above conditions are not true, decrement max holding by 1 and append a zero to returns column
        else:
            self.max_holding = self.max_holding - 1
            self.add_zeros()

    # -- Functions that merges the new columns created for the backtest into the dataframe, also resets the return series to empty list -- #
    def add_trade_cols(self):
        self.dmgt.df['returns'] = self.returns_series
        self.dmgt.df['holding'] = self.holding_series
        self.dmgt.df['direction'] = self.direction_series

        self.returns_series = []
        self.holding_series = []
        self.direction_series = []

    # -- Backtest heart -- #
    def run_backtest(self):
        # Signals generated from child class
        self.generate_signals()
        # Loop over dataframe
        for row in self.dmgt.df.itertuples():
            # If get a long signal and do not have open position -> open a long position
            if row.entry == 1 and self.open_pos is False:
                self.open_long(row.t_plus)
            # If get a short position and do not have open position -> open a short position
            elif row.entry == -1 and self.open_pos is False:
                self.open_short(row.t_plus)
            # Monitor open positions to see if any of the barriers have been touched, see function above
            elif self.open_pos:
                self.monitor_open_positions(row.close, row.Index)
            else:
                self.add_zeros()
        self.add_trade_cols()
    
    # -- Show a performance graph of the backtest -- #
    def show_performace(self):
        self.dmgt.df.returns.cumsum().plot()
        plt.title(f"Strategy results for {self.dmgt.timeframe} timeframe")
        plt.show()

    def save_backtest(self):
            strat_name = self.__class__.__name__
            tf = self.dmgt.timeframe
            self.dmgt.df.to_csv(f"../Backtests_Data/{strat_name}_{tf}.csv")