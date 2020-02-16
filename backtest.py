import pandas as pd
import datetime as dt
from prepare_data import CPrepareData
from train import CTrain
from predict import CPredict
import matplotlib.pyplot as plt
import numpy as np
from ann import CAnn
import os
import alphalens

class CBacktest():
	def __init__(self):
		self.traing_time_in_years = 3.5
		self.start_backtest_time = pd.to_datetime("2015-12-31", utc=True)
		self.end_backtest_time = pd.to_datetime("2019-03-30", utc=True)
		self.range_dates_for_backtest = pd.date_range(start=self.start_backtest_time, end=self.end_backtest_time, freq='1M', closed='right')
		self.init_balance = 100
		
	def run(self, train_overwriting_models=True):
		prepare = CPrepareData()
		train = CTrain()

		
		prepare.run(from_date=self.range_dates_for_backtest[0], to_date=self.range_dates_for_backtest[-1],
		            do_cleaning=False, remove_index=False)
		sxxp = prepare.marketdata_clean[prepare.marketdata_clean['ticker'] == "SXXP Index"]
		sxxp = pd.DataFrame({'SXXP Index': sxxp['Open'].values}, index=sxxp[train.settings.DATETIME_COL_NAME])
		
		train.read_data()
		train.create_encoders()
		
		predict = CPredict()
		
		predict.encoders = train.encoders.copy()
		
		for i, date in enumerate(self.range_dates_for_backtest):
			
			if i + 1 >= len(self.range_dates_for_backtest):
				break
			print(date)
			postfix = str(date.date())
			
			from_date = date - dt.timedelta(days=self.traing_time_in_years * 365)
			model_checkout_dir = CAnn.create_folder(train.settings.MODEL_DIR)
			
			_path = os.path.join(model_checkout_dir, 'model' + '_' + postfix + '.hdf5')
			
			if os.path.isfile(_path) is False or train_overwriting_models is True:
				prepare.run(from_date=from_date, to_date=date, do_cleaning=False, remove_index=True)
			
				
				train.read_data()
				train.run(postfix)
			from_date = date
			to_date = self.range_dates_for_backtest[i+1]
			
			prepare.run(from_date=from_date, to_date=to_date, do_cleaning=False, remove_index=True)
			
			predict.run_prediction(date, to_date)
			
			factor_data = self.generate_stats(predict)

			factor_data_2 = factor_data.reset_index()
			results = pd.DataFrame({'net': factor_data_2['1D'].values * factor_data_2['factor'].values, 'ticker': factor_data_2['asset'],
			                        'time': factor_data_2['date']})
			
			equity_stocks = pd.pivot_table(results, values='net', index=['time'], columns=['ticker'],
			                        aggfunc='sum')
			equity_stocks.sort_values(by='time', inplace=True)
			equity = (equity_stocks.sum(axis=1).cumsum(axis=0) + 1) * self.init_balance
			equity = pd.DataFrame({'Equity': equity}, index=equity_stocks.index)
			
			sharpe = self.calc_sharpe(equity['Equity'])
			profit = self.calc_profit(equity['Equity'])

			sharpe_sxxp = self.calc_sharpe(sxxp['SXXP Index'])
			profit_sxxp = self.calc_profit(sxxp['SXXP Index'])
			
			data = sxxp.merge(equity, 'outer', left_index=True, right_index=True)
			title = 'SXXP Sharpe = {:.02f}, SXXP Ann. Profit [%] = {:.02f}'. \
			            format(sharpe_sxxp, profit_sxxp)

			axes = data.dropna(how='any', axis=0).plot(fontsize=16, figsize=(18, 12), subplots=True, grid=True)
			axes[0].set_title(title, fontsize=18)
			axes[0].set_ylabel('Price', fontsize=16)
			axes[1].set_title('Portfolio Equity: Algo Sharpe = {:.02f}, Algo Ann. Profit [%] = {:.02f}'.
			            format(sharpe, profit), fontsize=18)
			axes[1].set_xlabel('Date', fontsize=16)
			axes[1].set_ylabel('Equity', fontsize=16)
			
			fig = plt.gcf()
			plt.tight_layout()
			fig.savefig('equity.png')
			del fig
			equity.to_csv('equity.csv')
			predict.prediction_df.to_csv('prediction.csv')

		factor_data.to_csv('factor_data.csv')
		#alphalens.tears.create_full_tear_sheet(factor_data)
			
	def calc_sharpe(self, series):
		return series.diff().rolling(250).sum().mean() / series.diff().rolling(250).sum().std()
	
	def calc_profit(self, series):
		return series.rolling(250).apply(lambda x: x[-1] - x[0]).mean() / self.init_balance * 100
	
	def generate_stats(self, predict):
		# Ingest and format data
		pricing = pd.DataFrame(
			{'open_price': predict.prediction_df['open_price'], 'date': predict.prediction_df['time'],
			 'asset': predict.prediction_df[predict.settings.CAT_COLS[0]]})
		my_factor = pd.DataFrame({'factor': predict.prediction_df['confidence'], 'date': predict.prediction_df['time'],
		                          'asset': predict.prediction_df[predict.settings.CAT_COLS[0]]})
		pricing['date'] = pd.to_datetime(pricing['date'], utc=True)
		pricing = pd.pivot_table(pricing, values='open_price', columns='asset', index='date', aggfunc='mean')
		print(pricing.head())
		my_factor['date'] = pd.to_datetime(my_factor['date'], utc=True)
		my_factor = pd.pivot_table(my_factor, values='factor', columns='asset', index='date', aggfunc='mean')
		my_factor = my_factor.stack(level='asset')
		my_factor = my_factor.reset_index()
		my_factor.set_index(['date', 'asset'], inplace=True)
		print(my_factor.head())
		factor_data = alphalens.utils.get_clean_factor_and_forward_returns(my_factor,
		                                                                   pricing, max_loss=1)

		# Run analysis
		
		return factor_data
		#alphalens.tears.create_returns_tear_sheet(factor_data)
	
	def renormalize_factor(self, predict):
		pass
	
if __name__ == '__main__':
    backtest = CBacktest()
    backtest.run(train_overwriting_models=False)