"""
Module responible for prediction by use of previously trained model. ANN model previously should be saved in
./models/model.hdf5 file in order to be properly read by this script. In order to use this class in other modules just use:

	prediction = CPredict()
	prediction.run()
	
Depending on settings, one can use kaggle environement to run script, or loading from file.
"""
import pandas as pd
import numpy as np
from ann import CAnn
from train import CTrain
import time
import os
import shutil
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

class CPredict(CTrain):
	def __init__(self):
		super().__init__()
		self.test = None
		self.pnl_df = None
		self.prediction_df = None
		
	def run_prediction(self, datetime_before_to_drop, datetime_after_to_drop):
		"""
		all process needed to make prediction by model
		:return:
		"""
		self.read_data()
		self.split_data()
		self.drop_data_for_test(datetime_before_to_drop=datetime_before_to_drop,
								datetime_after_to_drop=datetime_after_to_drop)
		self.encoding()
		self.scaling_numerical_data()
		nn = CAnn(cat_cols=self.cat_cols, embed_sizes=self.embed_sizes, log=self.log)
		nn.create_folder(self.settings.MODEL_DIR)
		self.model = nn.create(len(self.settings.NUM_COLS))
		self.log.info(self.model.summary())
		self.model = nn.load_model(self.model, str(datetime_before_to_drop.date()))

		if self.settings.DATA_FROM_FILE > 0:
			self.read_submission_sample()
			
			_dates = self.test[self.settings.DATETIME_COL_NAME].unique()
			_dates = np.sort(_dates)
			
			self.data_for_prediction = ((self.test[self.test[self.settings.DATETIME_COL_NAME] == d], d,
										self.submission_sample[self.submission_sample[self.settings.DATETIME_COL_NAME] == d]) for d in _dates)
		else:
			self.log.info("Option with DATA_FROM_FILE == 0 is inactive")
			return
		
		self.predict(self.data_for_prediction)
		
		self.save_submission()

	def drop_data_for_test(self, datetime_before_to_drop, datetime_after_to_drop):
		"""
		creating test data
		:param datetime_before_to_drop: datetime
		:param datetime_after_to_drop: datetime
		:return:
		"""
		self.log.info("Removing data before {} and after {} ".format(str(datetime_before_to_drop),
																	 str(datetime_after_to_drop)))
	
		self.test = self.train.loc[(self.train[self.settings.DATETIME_COL_NAME] > datetime_before_to_drop) &
								   (self.train[self.settings.DATETIME_COL_NAME] <= datetime_after_to_drop)]
		
		def to_date(x):
			return x._date_repr
			
		self.test[self.settings.DATETIME_COL_NAME] = self.test[self.settings.DATETIME_COL_NAME].apply(to_date)
		self.test[self.settings.DATETIME_COL_NAME] = pd.to_datetime(self.test[self.settings.DATETIME_COL_NAME])
	
		
	def read_submission_sample(self):
		"""
		reading file with sample submission
		:return: bool
		"""
		_path_to_file = os.path.realpath(self.settings.RAW_DATA_DIR)
		_path_to_file =os.path.join(_path_to_file, "submission_sample.csv")
		if os.path.isfile(_path_to_file):
			self.submission_sample = pd.read_csv(_path_to_file)
			self.submission_sample[self.settings.DATETIME_COL_NAME] = pd.to_datetime(self.submission_sample['time'])
			return True
		
		self.log.error("No submission_sample file {}".format(_path_to_file))
		return False
	
	def predict(self, days):
		"""
		making prediction using loaded model fom model.hdf5
		:param days: generator returning (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
		:return:
		"""
		n_days = 0
		max_n_days = len(self.test[self.settings.DATETIME_COL_NAME].unique())
		prep_time = 0
		prediction_time = 0
		self.predictions_df = None
		packaging_time = 0
		predicted_confidences = np.array([])
		X_test = {}

		yesterday_preds = None
		for (market_obs_df, current_day, _) in days:
			n_days += 1
			if n_days % 100 == 0:
				self.log.info("done {}%".format(round(n_days * 100.0 / max_n_days, 2)))
			
			prices = market_obs_df['Open'].copy()
			
			t = time.time()
		
			market_obs_df['assetCode_encoded'] = market_obs_df[self.cat_cols[0]].astype(str).\
				apply(lambda x: self.encode(self.encoders[0], x))
		
			market_obs_df[self.num_cols] = market_obs_df[self.num_cols].fillna(0)
			market_obs_df[self.num_cols] = self.scaler.transform(market_obs_df[self.num_cols])
			X_num_test = market_obs_df[self.num_cols].values
			X_test['num'] = X_num_test
			X_test[self.settings.CAT_COLS[0]] = market_obs_df.loc[:, 'assetCode_encoded'].values
		
			prep_time += time.time() - t
		
			t = time.time()
			market_prediction = self.model.predict(X_test)[:, 0] * 2 - 1
			#market_prediction = np.array([p if p > 0 else 0 for p in market_prediction], dtype='float')
			market_prediction = market_prediction / np.sum(np.abs(market_prediction[market_prediction != 0]))

			prediction_time += time.time() - t
		
			t = time.time()
			preds = pd.DataFrame({'time': [current_day] * len(market_obs_df),
			                      self.settings.CAT_COLS[0]: market_obs_df[self.settings.CAT_COLS[0]],
			                      'open_price': prices,
			                      'confidence': market_prediction}).fillna(0)
			
			if yesterday_preds is None:
				yesterday_preds = preds.copy()
			
			if self.pnl_df is None:
				self.prediction_df = preds
				self.pnl_df = pd.DataFrame(columns=['pnl', 'cost', 'net', 'time', 'ticker'])
			else:
				
				self.prediction_df = self.prediction_df.append(preds)
				market_obs_df['confidence'] = market_prediction
				
				pnl = market_obs_df[self.settings.BACKTEST_COL].values * preds['confidence'].values
				
				costs = self.settings.TRADE_COST_PERCENT / 100 * np.abs(preds['confidence'].values -
				                                                        yesterday_preds['confidence'].values)
				
				self.pnl_df = self.pnl_df.append(pd.DataFrame(data=np.vstack([pnl.T, costs.T, pnl-costs.T,
																	market_obs_df[self.settings.DATETIME_COL_NAME].astype('str').values.T,
														  market_obs_df[self.settings.CAT_COLS[0]].values.T]).T,
													columns=['pnl', 'cost', 'net', 'time', 'ticker']))
				
			yesterday_preds = preds.copy()
			packaging_time += time.time() - t
		
		self.pnl_df = self.pnl_df.ffill()
	
		total = prep_time + prediction_time + packaging_time
		self.log.info(f'Preparing Data: {prep_time:.2f}s')
		self.log.info(f'Making Predictions: {prediction_time:.2f}s')
		self.log.info(f'Packing: {packaging_time:.2f}s')
		self.log.info(f'Total: {total:.2f}s')

	def create_folder(self, folder_name):
		"""
		creating directory with given name
		:param folder_name: string
		:return:
		"""
		_path = os.path.realpath(folder_name)
		if os.path.isdir(_path) == False:
			try:
				os.mkdir(_path)
			except Exception as ex:
				print("Could not create directory: {}".format(_path))
				raise ex
		self.model_checkout_dir = _path
		
	def save_submission(self):
		"""
		saving submission to file
		:return:
		"""
		
		self.create_folder(self.settings.SUBMISSION_DIR)
		_path_to_file = os.path.join(self.settings.SUBMISSION_DIR, "pnls.csv")
		self.pnl_df.to_csv(_path_to_file, index=True)
		_path_to_file = os.path.join(self.settings.SUBMISSION_DIR, "predictions.csv")
		self.prediction_df.to_csv(_path_to_file, index=False)
		if self.settings.DATA_FROM_FILE == 0:
			self.env.write_submission_file()
			shutil.copy("submission.csv", _path_to_file)
			os.remove("submission.csv")
	
	
if __name__ == '__main__':
	prediction = CPredict()
	prediction.run()