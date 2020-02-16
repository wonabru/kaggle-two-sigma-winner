"""
Module which generate Artificial Neural Network architecture. This module is used in training module and predicting one.
"""
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization
from keras import regularizers
from keras import initializers
from keras.layers import Dropout
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import os

class CAnn:
	def __init__(self, cat_cols, embed_sizes, log):
		self.r_l1 = 0
		self.r_l2 = 0
		self.dropout = 0.4
		self.cat_cols = cat_cols
		self.embed_sizes = embed_sizes
		self.log = log
		
	@staticmethod
	def custom_objective(y_true, y_pred):
		"""
		Custom objective loss function. In general it maximizes Sharp ratio.
		:param y_true: tensorflow array
		:param y_pred: tensorflow array
		:return: tensorflow float
		"""
		x = y_true * (2 * y_pred - 1)
		return 100 - (K.sum(x) / (K.std(x) + 0.001))

	def create(self, nb_numerical_inputs):
		"""
		creating architecture of our ANN model
		:return: tensorflow object
		"""
		categorical_inputs = []
		for cat in self.cat_cols:
			categorical_inputs.append(Input(shape=[1], name=cat))
		
		categorical_embeddings = []
		for i, cat in enumerate(self.cat_cols):
			categorical_embeddings.append(Embedding(self.embed_sizes[i], 10)(categorical_inputs[i]))

		categorical_logits = Flatten()(categorical_embeddings[0])
		categorical_logits = Dense(8,activation='relu', kernel_initializer=initializers.glorot_uniform(seed=23),
		                           kernel_regularizer=regularizers.l2(self.r_l2),
						activity_regularizer=regularizers.l1(self.r_l1))(categorical_logits)
		categorical_logits = Dropout(self.dropout)(categorical_logits)
		
		numerical_inputs = Input(shape=(nb_numerical_inputs,), name='num')
		numerical_logits = numerical_inputs
		numerical_logits = BatchNormalization()(numerical_logits)
		
		numerical_logits = Dense(32,activation='relu', kernel_initializer=initializers.glorot_uniform(seed=23),
		                         kernel_regularizer=regularizers.l2(self.r_l2),
						activity_regularizer=regularizers.l1(self.r_l1))(numerical_logits)
		numerical_logits = Dropout(self.dropout)(numerical_logits)
		
		numerical_logits = Dense(16,activation='relu', kernel_initializer=initializers.glorot_uniform(seed=23),
		                         kernel_regularizer=regularizers.l2(self.r_l2),
						activity_regularizer=regularizers.l1(self.r_l1))(numerical_logits)
		numerical_logits = Dropout(self.dropout)(numerical_logits)
		
		numerical_logits = Dense(8,activation='relu', kernel_initializer=initializers.glorot_uniform(seed=23),
		                         kernel_regularizer=regularizers.l2(self.r_l2),
						activity_regularizer=regularizers.l1(self.r_l1))(numerical_logits)
		numerical_logits = Dropout(self.dropout)(numerical_logits)
		
		logits = Concatenate()([numerical_logits,categorical_logits])
		logits = Dense(16,activation='relu', kernel_initializer=initializers.glorot_uniform(seed=23),
		               kernel_regularizer=regularizers.l2(self.r_l2),
						activity_regularizer=regularizers.l1(self.r_l1))(logits)
		logits = Dropout(self.dropout)(logits)
		
		out = Dense(1, activation='sigmoid')(logits)
		
		model = Model(inputs = categorical_inputs + [numerical_inputs], outputs=out)
		model.compile(optimizer='adam',loss=self.custom_objective)
		
		return model
	
	def train_model(self, model, X_train, r_train, X_valid, r_valid, postfix):
		"""
		traing ANN model
		:param model: tensorflow object
		:param X_train: list of floats
		:param r_train: list of floats
		:param X_valid: list of floats
		:param r_valid: list of floats
		:return:
		"""
		_path = os.path.join(self.model_checkout_dir, 'model' + '_' + postfix + '.hdf5')

		check_point = ModelCheckpoint(_path, verbose=True, save_best_only=True)
		early_stop = EarlyStopping(patience=3, verbose=True)
		model.fit(X_train, r_train,
				  validation_data=(X_valid, r_valid),
				  epochs=1,
				  verbose=False,
				  callbacks=[early_stop, check_point])
		
	def load_model(self, model, postfix):
		"""
		loading ANN model from file
		:param self: CAnn class object
		:param model: tensorflow object
		:return: tensorflow object
		"""
		_path_to_model = os.path.join(self.model_checkout_dir, 'model' + '_' + postfix + '.hdf5')
		model.load_weights(_path_to_model)
		return model
	
	def score(self, model, X_valid, r_valid, u_valid, d_valid, postfix):
		"""
		calculating final score using validation data
		:param model: tensorflow object
		:param X_valid: list of floats
		:param r_valid: list of floats
		:param u_valid: list of floats
		:param d_valid: list of floats
		:return: float
		"""
	
		model = self.load_model(model, postfix)
		confidence_valid = model.predict(X_valid)[:, 0] * 2 - 1
	
		r_valid = r_valid.clip(-1, 1)  # get rid of outliers. Where do they come from??
		x_t_i = confidence_valid * r_valid * u_valid
		data = {'day': d_valid, 'x_t_i': x_t_i}
		df = pd.DataFrame(data)
		x_t = df.groupby('day').sum().values.flatten()
		mean = np.mean(x_t)
		std = np.std(x_t)
		score_valid = mean / std
		
		return score_valid
	
	@classmethod
	def create_folder(self, folder_name):
		"""
		creating director for given name
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
		return _path