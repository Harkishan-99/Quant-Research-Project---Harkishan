import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from tqdm.notebook import tqdm

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

        :return : None
        """
        raise NotImplementedError('Subclass must implement this!')
        
    def run(self)->None:
        """
        Run the vectorized backtest.
        """
        self.signals = self.get_signals()
        returns = (1+self.PL).pct_change()[1:]
        pnl = self.PL.diff()[1:]
        self.returns_strat = self.signals * returns
        self.pnl_strat = self.signals * pnl
  
    def get_summary(self, show=True)->None:
        """
        Calculate the backtest summary results.

        :param show: (bool) whether to print the report 
        :return : None
        """
        #Calculate cumulative P&L
        self.equity_pnl = self.pnl_strat.cumsum()
        #Calculate the sharpe ratio
        self.sharpe = np.nanmean(self.returns_strat)/np.nanstd(self.returns_strat)*np.sqrt(252*375)
        #Calculate running maximum
        running_max = self.equity_pnl.cummax()
        #Calculate drawdown
        drawdowns = self.equity_pnl - running_max
        #Maximum Drawdown
        max_dd = drawdowns.min()
        entries = self.signals.diff()[1:]
        if show:
            #Print Metrics
            print("                   Results              ")
            print("-------------------------------------------")
            print("%14s %21s" % ('statistic', 'value'))
            print("-------------------------------------------")
            print("%20s %20.2f" % ("Absolute P&L :", self.equity_pnl[-1]))
            print("%20s %20.2f" % ("Sharpe Ratio :", self.sharpe))
            print("%20s %20.2f" % ("Max. Drawdown P&L:", round(max_dd, 2)))
            #print("%20s %20.2f" % ("Total Trades :", sum(abs(entries))))
            #Plots
            x = self.equity_pnl.index
            #fig, axs = plt.subplots(4, figsize=(16, 15), height_ratios=[4, 3, 4, 4])
            fig, axs = plt.subplots(3, figsize=(16, 12), height_ratios=[4, 4, 4])
            fig.suptitle('Backtest Report', fontweight="bold")
            axs[0].plot(x, self.spread.values, color='#aec6cf')
            axs[0].title.set_text("P/L")
            axs[0].grid()
            # axs[1].plot(x, self.signals.values, color='#aec6cf')
            # axs[1].title.set_text("Positions")
            # axs[1].grid()
            axs[1].plot(x, self.equity_pnl.values, color='#77dd77')
            axs[1].title.set_text("Strategy Equity Curve : P&L")
            axs[1].grid()
            axs[2].fill_between(x, drawdowns.values, color='#ff6961', alpha=0.5)
            axs[2].title.set_text("Drawdowns : P&L")
            axs[2].grid()
            plt.show()


class Optimizer:
    def __init__(self, strategy:VectorizedBT, df:pd.DataFrame):
        self.strategy = strategy
        self.df = df
        
    def search(self, params:dict={}, category:str='sharpe')->None:
        """
        Run grid search over the parameters.
        Returns the optimal parameters for 
        the strategy that maximize the 
        category.
        
        :param category :(str) Use either 'sharpe' or 'pnl'
        :param params :(dict) search dictionary for the 
                        strategy parameters. Make sure 
                        the keys are the arguments of strategy.
        :return : None
        """
        keys = list(params.keys())
        search_space = [dict(zip(keys, values)) for values in product(*params.values())]
        self.best_category_val = -np.inf
        self.best_param = None
        for search_param in tqdm(search_space):
            this_strat = self.strategy(self.df, **search_param)
            this_strat.run()
            this_strat.get_summary(False)
            val = None
            if category=='sharpe':
                val = this_strat.sharpe
            elif category=='pnl':
                val = this_strat.equity_pnl
            else:
                raise ValueError("Invalid Category.")
                
            if val > self.best_category_val:
                self.best_category_val = val
                self.best_param = search_param
        print(f"Best {category} score achived : {self.best_category_val}\n")
        print(f"Best parameters for the above score : {self.best_param}") 