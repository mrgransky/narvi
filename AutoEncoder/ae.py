from datetime import datetime
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard

import numpy as np 
from tensorflow import set_random_seed
import os

def seedy(s):
    np.random.seed(s)
    set_random_seed(s)

class AutoEncoder:
	def __init__(self, encoding_dim=3):
		self.encoding_dim = encoding_dim
		r = lambda: np.random.randint(1, 3)
		self.x = np.array([[r(),r(),r()] for _ in range(1000)])
		print(self.x)

	def _encoder(self):
		inputs = Input(shape=(self.x[0].shape))
		encoded = Dense(self.encoding_dim, activation='relu')(inputs)
		model = Model(inputs, encoded)
		self.encoder = model
		return model

	def _decoder(self):
		inputs = Input(shape=(self.encoding_dim,))
		decoded = Dense(3)(inputs)
		model = Model(inputs, decoded)
		self.decoder = model
		return model

	def encoder_decoder(self):
		ec = self._encoder()
		dc = self._decoder()
		
		inputs = Input(shape=self.x[0].shape)
		ec_out = ec(inputs)
		dc_out = dc(ec_out)
		model = Model(inputs, dc_out)
		
		self.model = model
		return model

	def fit(self, batch_size=10, epochs=300):
		self.model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
		now = datetime.now() # current date and time
		date_time = now.strftime("%d%m%Y%H%M%S")
		print "\nTime: \t", date_time
		NAME = "AE_{}".format(int(date_time))
		log_dir = "logs/{}".format(NAME)
		tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
		self.model.fit(self.x, self.x, epochs=epochs, 
							batch_size=batch_size, callbacks=[tb], validation_split=0.1)

	def save(self):
		if not os.path.exists(r'./weights'):
			os.mkdir(r'./weights')
		
		self.encoder.save(r'./weights/encoder_weights.h5')
		self.decoder.save(r'./weights/decoder_weights.h5')
		self.model.save(r'./weights/ae_weights.h5')
        

if __name__ == '__main__':
    seedy(2)
    ae = AutoEncoder(encoding_dim=2)
    ae.encoder_decoder()
    ae.fit(batch_size=50, epochs=300)
    ae.save()
