import numpy as np
import matplotlib.pyplot as plt

class VectorizedBT(object):
    """
    A class to backtest trading signals 
    using vectorized backtesting method.
    """
    def __init__(self):
        pass
        
    def get_signals(self):
        """
        Implement the strategy signals here.
        """
        raise NotImplementedError('Subclass must implement this!')
        
    def run(self):
        self.signals = self.get_signals()
        returns = (1+self.PL).pct_change()[1:]
        pnl = self.PL.diff()[1:]
        self.returns_strat = self.signals * returns
        self.pnl_strat = self.signals * pnl
  
    def get_perf_stats(self):
        #Calculate cumulative P&L
        self.equity_pnl = self.pnl_strat.cumsum()
        #Calculate the sharpe ratio
        sharpe = np.nanmean(self.returns_strat)/np.nanstd(self.returns_strat)*np.sqrt(252*375)
        #Calculate running maximum
        running_max = self.equity_pnl.cummax()
        #Calculate drawdown
        drawdowns = self.equity_pnl - running_max
        #Maximum Drawdown
        max_dd = drawdowns.min()
        entries = self.signals.diff()[1:]
        #Print Metrics
        print("                   Results              ")
        print("-------------------------------------------")
        print("%14s %21s" % ('statistic', 'value'))
        print("-------------------------------------------")
        print("%20s %20.2f" % ("Absolute P&L :", self.equity_pnl[-1]))
        print("%20s %20.2f" % ("Sharpe Ratio :", sharpe))
        print("%20s %20.2f" % ("Max. Drawdown P&L:", round(max_dd, 2)))
        #print("%20s %20.2f" % ("Total Trades :", sum(abs(entries))))
        #Plots
        x = self.equity_pnl.index
        fig, axs = plt.subplots(4, figsize=(16, 15), height_ratios=[4, 3, 4, 4])
        fig.suptitle('Backtest Report', fontweight="bold")
        axs[0].plot(x, self.spread.values, color='orange')
        axs[0].title.set_text("P/L")
        axs[0].grid()
        axs[1].plot(x, self.signals.values, color='#aec6cf')
        axs[1].title.set_text("Positions")
        axs[1].grid()
        axs[2].plot(x, self.equity_pnl.values, color='#77dd77')
        axs[2].title.set_text("Strategy Equity Curve : P&L")
        axs[2].grid()
        axs[3].fill_between(x, drawdowns.values, color='#ff6961', alpha=0.5)
        axs[3].title.set_text("Drawdowns : P&L")
        axs[3].grid()
        plt.show()