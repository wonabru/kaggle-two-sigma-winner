"""
In general the module is responsible for training data using ANN model. In order to use this class in other module:

		train_nn = CTrain()
		train_nn.run()

The sequence for whole procedure of training one can see in function 'run()'
"""
import pandas as pd
from read_settings import CSettings
from logger import CLogger
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ann import CAnn
import numpy as np


class CTrain:
	def __init__(self):
		self.settings = CSettings()
		self.train = None
		self.cat_cols = self.settings.CAT_COLS
		self.num_cols = self.settings.NUM_COLS
		self.log = CLogger(self.settings.LOGS_DIR)
		self.log.info(self.settings.print())

	def read_data(self):
		"""
		Reading data from file saved previously by prepare_data.py script
		:return: bool
		"""
		_path_to_train_file = os.path.realpath(self.settings.TRAIN_DATA_CLEAN_PATH)
		if os.path.isfile(_path_to_train_file):
			self.log.info("Loading data from file {}. This may take a few minutes...".format(_path_to_train_file))
			self.train = pd.read_csv(_path_to_train_file)
			self.log.info("Parsing dates...")
			self.train[self.settings.DATETIME_COL_NAME] = pd.to_datetime(self.train[self.settings.DATETIME_COL_NAME])
			self.log.info(str(self.train.describe()))
			return True
		self.log.error("No train data {}".format(self.settings.TRAIN_DATA_CLEAN_PATH))
		return False

	def encode(self, encoder, x):
		"""
		Replacing strings by integers
		:param self: CTrain class object
		:param encoder: int
		:param x: string
		:return:
		"""
		len_encoder = len(encoder)
		try:
			id = encoder[x]
		except KeyError:
			id = len_encoder
		return id
	
	def create_encoders(self):
	
		self.encoders = [{} for cat in self.cat_cols]
		
		for i, cat in enumerate(self.cat_cols):
			self.log.info('encoding {} ... '.format(cat))
			self.encoders[i] = {l: id for id, l in enumerate(self.train.loc[:, cat].
															 astype(str).unique())}
	
	def encoding(self):
		"""
		Encoding category features
		:return:
		"""
		
		for i, cat in enumerate(self.cat_cols):
			self.log.info('encoding {} ... '.format(cat))
			self.train[cat] = self.train[cat].astype(str).apply(lambda x: self.encode(self.encoders[i], x))
			self.log.info('Done')
	
		self.embed_sizes = [len(encoder) + 1 for encoder in self.encoders]  # +1 for possible unknown assets
	
	def scaling_numerical_data(self):
		"""
		Scaling numerical features
		:return:
		"""
		self.train[self.num_cols] = self.train[self.num_cols].fillna(0)
		self.log.info('Scaling numerical columns')
	
		self.scaler = StandardScaler()
		self.scaler.fit(self.train[self.num_cols])
		self.log.info('Scaling done')
		
	def split_data(self):
		"""
		Spliting data to train and validation sets
		:return:
		"""
		self.log.info('Spliting to train and test data')
		self.train_indices, self.val_indices = train_test_split(self.train.index.values,
																 test_size=0.4, random_state=42, shuffle=True)

		self.log.info('Spliting done')

	def get_input(self, market_train, indices):
		"""
		generating features for ANN input
		:param market_train: pandas. DataFrame
		:param indices: list of int
		:return: tuple of lists
		"""
		X_num = market_train.loc[indices, self.num_cols].values
		X = {'num': X_num}
		for cat in self.cat_cols:
			X[cat] = market_train.loc[indices, self.cat_cols].values
		y = (market_train.loc[indices, self.settings.TARGET_COL] >= 0).values
		r = market_train.loc[indices, self.settings.TARGET_COL].values
		u = np.array([1] * len(indices))
		d = market_train.loc[indices, self.settings.DATETIME_COL_NAME].dt.date
		return X, y, r, u, d
	
	def run(self, postfix):
		"""
		Sequence of all procedures. This function should be run to train model after class initialization.
		It uses 'CAnn' class, which create architecture of our NN model.
		:return:
		"""
		#self.read_data()
		# self.create_encoders()
		self.split_data()
		
		
		self.encoding()
		
		self.scaling_numerical_data()
		self.train[self.num_cols] = self.scaler.transform(self.train[self.num_cols])
		
		nn = CAnn(cat_cols=self.cat_cols, embed_sizes=self.embed_sizes, log=self.log)
		nn.create_folder(self.settings.MODEL_DIR)
		self.model = nn.create(len(self.settings.NUM_COLS))
		self.log.info(self.model.summary())

		self.log.info('Transforming data')
		self.X_train, self.y_train, self.r_train, self.u_train, self.d_train = self.get_input(self.train, self.train_indices)
		self.X_valid, self.y_valid, self.r_valid, self.u_valid, self.d_valid = self.get_input(self.train, self.val_indices)
		self.log.info('Transforming data done')
		
		nn.train_model(self.model, self.X_train, self.r_train, self.X_valid, self.r_valid, postfix)
		
		score = nn.score(self.model, self.X_valid, self.r_valid, self.u_valid, self.d_valid, postfix)
		
		self.log.info("Test score = {}".format(score))
		
if __name__ == '__main__':
	train_nn = CTrain()
	train_nn.run()