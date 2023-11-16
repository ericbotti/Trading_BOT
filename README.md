# Eric Bottinelli Bachelor's Thesis

Accurately predicting asset values in a financial market is an extremely
tough endeavor. Pricing is determined by several factors, which
are often random or unpredictable, characteristic which constitutes one
of the difficulties of this task. This thesis first studies the implementation
of an artificial neural network (ANN) to forecast a new type of asset,
Bitcoin perpetual futures (BPF), a financial instrument used to speculate
on the increase or decrease of Bitcoin’s price. The goal is not only
to forecast the price of this asset but also to develop a high-frequency
trading algorithm capable of trading the rise or fall of the underlying
asset in the real market. To achieve this goal, a recurrent neural network
(RNN) model employing the long short-term memory (LSTM)
architecture was created and trained utilizing 30 features, including
26 technical indicators, and a dataset consisting of historical data collected
every minute for 802022 periods. A trading strategy including
a decision-making system that generates signals to open long or short
BPF positions was developed and backtested with the outcomes compared
to those of three conventional strategies: Momentum RSI, higher
highs and lower lows (HHLL), and Buy-and-Hold. In order to engage
in real-money trading, an algorithm capable of receiving every minute
data from Deribit, a perpetual futures exchange, and using the neural
network established to anticipate the price of the following minute
was developed. In addition to the previously developed strategy functionalities,
the ability to automatically purchase and sell on Deribit’s
platform and verify open positions has been introduced. The results
from nine trading days were encouraging, as the profit corresponds to
2.63% after 19 performed trades, of which 13 long and 6 short. Furthermore,
the backtesting results demonstrated remarkable performance,
with the LSTM approach outperforming the three standard methods
with an aggregate return of 9.75% in 112 days and a compound annual
growth rate (CAGR) of 35.75%, a Sharpe ratio of 1.70, a Calmar ratio of
4.754, and a maximum drawdown (MDD) of 6.8%. The trained neural
network achieved a mean squared error (MSE) of 6732, mean absolute
error (MAE) of 55.9, and mean absolute percentage error (MAPE) of
0.134. With their respective limitations, the findings suggest that the
developed neural network model has great potential for predicting BPF
prices and that the algorithm seems to be profitable on both historical
and current data.
