"""
This script prepare the data by cleaning it from outliers in 'open' and 'close' prices. The way of using it:
	
	python3 prepare_data.py
	
All important parameters are in SETTINGS.json.
In the case of using script in kaggle environement, one should set "DATA_FROM_FILE": 0
In the opposite situation, just to take data from file, set "DATA_FROM_FILE": 1, and set also "DATA_FILENAME".
This file will be search in "RAW_DATA_DIR" directory.
The clean data are saved in "TRAIN_DATA_CLEAN_PATH" file
"""
import pandas as pd
import numpy as np
from read_settings import CSettings
from logger import CLogger
import os


class CPrepareData:
	def __init__(self):
		self.settings = CSettings()
		self.raw_data = None
		self.marketdata = None
		self.marketdata_clean = None
		self.file_name = self.settings.DATA_FILENAME
		self.log = CLogger(self.settings.LOGS_DIR)
		self.log.info(self.settings.print())
	
	def read_data(self, file_name):
		"""
		reading data from file
		:param file_name: string
		:return:
		"""
		if self.raw_data is None:
			_path_to_file = os.path.realpath(self.settings.RAW_DATA_DIR)
			_path_to_file = os.path.join(_path_to_file, file_name)
			if os.path.isfile(_path_to_file):
				self.log.info("Loading data from file {}. This may take a few minutes...".format(_path_to_file))
				self.raw_data = pd.read_csv(_path_to_file)
				self.log.info("Parsing dates...")
				self.raw_data[self.settings.DATETIME_COL_NAME] = pd.to_datetime(self.raw_data[self.settings.DATETIME_COL_NAME], utc=True)
				self.raw_data = self.raw_data.iloc[:, :self.settings.NUMBER_OF_INSTRUMENTS]
				self.log.info(str(self.raw_data.describe()))
				return True
		else:
			return True
		self.log.error("No input data {} found in directory {}".format(file_name, self.settings.RAW_DATA_DIR))
		return False

	def write_data(self, data, data_clean_path):
		_path_to_file = os.path.realpath(data_clean_path)
		self.log.info("Saving data to file {} ...".format(_path_to_file))
		data.to_csv(_path_to_file, index=False)
		self.log.info("Data saved")
		
	def __replace_NaN_with_mean(self, data):
		mean = np.nanmean(data)
		indicies_NaN = ~np.isfinite(data)
		data[indicies_NaN] = mean
		return data
	
	def __replace_outliers_with_zero(self, data, range_in_sd=5):
		mean = np.nanmean(data)
		sd = np.nanstd(data) * range_in_sd
		data[(data > mean + sd) | (data < mean - sd)] = 0
		return data
	
	def clean_data(self, do_cleaning, remove_index=True):
		"""
		The main function, which clean self.marketdata and next save it in self.marketdata_clean.
		Cleaning data here means repalcing outlier prices to the mean from all traning data set.
		The function check if difference between 'open' and 'close' prices are larger than 200% or smaller than -50%,
		then replacing 'open' or 'close' price depending which price is outlier.
		:return:
		"""
		self.log.info("Start cleaning data...")
		self.marketdata_clean = self.marketdata.replace([np.Inf, -np.Inf], np.NaN)

		for col in self.settings.RETURN_COLS:
			self.marketdata_clean[col] = self.__replace_outliers_with_zero(self.marketdata_clean[col], 1000)
			
		if remove_index is True:
			self.marketdata_clean = self.marketdata_clean[
			self.marketdata_clean[self.settings.CAT_COLS[0]] != self.settings.INDEX]
		
		self.marketdata_clean[self.settings.TARGET_COL] = self.marketdata_clean[self.settings.TARGET_COL].fillna(0)
		self.marketdata_clean[self.settings.BACKTEST_COL] = self.marketdata_clean[self.settings.BACKTEST_COL].fillna(0)
		
		if do_cleaning is False:
			
			self.marketdata_clean = self.marketdata_clean.ffill()
			return
			
		self.marketdata_clean['assetCode_mean_open'] = self.marketdata_clean.groupby(self.settings.CAT_COLS)['Open'].transform(np.nanmean)
		self.marketdata_clean['assetCode_mean_close'] = self.marketdata_clean.groupby(self.settings.CAT_COLS)['Close1'].transform(np.nanmean)
		self.marketdata_clean = self.marketdata_clean.ffill()
		self.marketdata_clean['close_to_open'] = self.marketdata_clean['Close1'] / self.marketdata_clean['Open']

		for col in self.settings.NUM_COLS:
			self.marketdata_clean[col] = self.__replace_NaN_with_mean(self.marketdata_clean[col])
		
	
		# if open price is too far from mean open price for this company, replace it. Otherwise replace close price.
		_data = self.marketdata_clean.loc[self.marketdata_clean['close_to_open'] >= 5]
		_data_len = len(_data)
		self.log.info("Replacing outlier prices above 200% with mean ... amount:" + str(_data_len))
		if _data_len > 0:
			for j, (index, row) in enumerate(_data.iterrows()):
				if np.abs(row['assetCode_mean_open'] - row['Open']) > np.abs(row['assetCode_mean_close'] - row['Close1']):
					self.marketdata_clean.loc[index, 'Open'] = row['assetCode_mean_open']
				else:
					self.marketdata_clean.loc[index, 'Close1'] = row['assetCode_mean_close']
				if j % 1000 == 0:
					self.log.info("done {}%".format(round(j * 100.0 / _data_len, 2)))
				
		_data = self.marketdata_clean.loc[self.marketdata_clean['close_to_open'] <= 0.05]
		_data_len = len(_data)
		self.log.info("Replacing outlier prices below -50% with mean ... amount:" + str(_data_len))
		if _data_len > 0:
			for j, (index, row) in enumerate(_data.iterrows()):
				if np.abs(row['assetCode_mean_open'] - row['Open']) > np.abs(row['assetCode_mean_close'] - row['Close1']):
					self.marketdata_clean.loc[index, 'Open'] = row['assetCode_mean_open']
				else:
					self.marketdata_clean.loc[index, 'Close1'] = row['assetCode_mean_close']
				if j % 1000 == 0:
					self.log.info("done {}%".format(round(j * 100.0 / _data_len, 2)))
				
		self.log.info("Done cleaning data.")
		self.log.info(str(self.marketdata_clean.describe()))
	
	def create_folder(self, folder_name):
		"""
		This function creates directory with given name 'folder_name'
		:param folder_name: string
		:return:
		"""
		_path = os.path.realpath(folder_name)
		_path = os.path.split(_path)[0]
		if os.path.isdir(_path) == False:
			try:
				os.mkdir(_path)
			except Exception as ex:
				self.log.error("Could not create directory: {}".format(_path))

	def drop_data(self, datetime_before_to_drop, datetime_after_to_drop):
		"""
		Removing data which are before 'datetime_before_to_drop'
		:param datetime_before_to_drop: datetime
		:return:
		"""
		self.log.info("Removing data before {}".format(str(datetime_before_to_drop)))
		self.marketdata = self.raw_data.loc[(self.raw_data[self.settings.DATETIME_COL_NAME] >= datetime_before_to_drop) &
									(self.raw_data[self.settings.DATETIME_COL_NAME] < datetime_after_to_drop)]
	
	def run(self, from_date, to_date, do_cleaning, remove_index):
		"""
		Function which run all seperate functions, causing reading data, droping unwanted data, cleaning and
		writing clean data to file.
		:return:
		"""
		if self.settings.DATA_FROM_FILE > 0 and self.read_data(self.file_name):
			pass
		else:
			self.log.error("There is a problem with loading data")
		
		self.drop_data(from_date, to_date)
		
		self.clean_data(do_cleaning=do_cleaning, remove_index=remove_index)
		
		self.create_folder(self.settings.TRAIN_DATA_CLEAN_PATH)
		self.write_data(self.marketdata_clean, self.settings.TRAIN_DATA_CLEAN_PATH)
		

if __name__ == '__main__':
	
	prepare_data = CPrepareData()
	prepare_data.run()