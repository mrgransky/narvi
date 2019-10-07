from datetime import datetime
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time, sys, os

if len(sys.argv) != 3:
	print "\nSYNTAX: \n\npython " + sys.argv[0] + " [PATH/2/features (X)] [PATH/2/labels (y)] \n"
	sys.exit()

features = sys.argv[1]
labels = sys.argv[2]

"""
dense_layers 	= [0]
conv_layers 	= [3]
layer_sizes 	= [64,	256,	512,	1024	]
"""

dense_layers 	= [0]
conv_layers 	= [3]
layer_sizes 	= [32, 64, 128, 256, 512, 1024]


num_epoch = 10
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

pickle_in = open(str(features),"rb")
X = pickle.load(pickle_in)

print "\n\nX shape = ", X.shape[1:]

pickle_in = open(str(labels),"rb")
y = pickle.load(pickle_in)

X = X/255.0 # [0 - 1]

for dense_layer in dense_layers:
	for layer_size in layer_sizes:
		for conv_layer in conv_layers:
			now = datetime.now() # current date and time
			date_time = now.strftime("%d%m%Y%H%M%S")
			print "\nTime:", date_time
			#NAME = "{}_CONV_{}_NODE_{}_Dense_{}".format(conv_layer, layer_size, 
			#															dense_layer, int(time.time()))
			
			NAME = "{}_CONV_{}_NODE_{}_Dense_{}".format(conv_layer, layer_size, 
																		dense_layer, int(date_time))
			
			#print(NAME)
			
			tb = TensorBoard(log_dir='logs/{}'.format(NAME))
			
			model = Sequential()
			
			model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2, 2)))
			
			for l in range(conv_layer-1):
				model.add(Conv2D(layer_size, (3, 3)))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2, 2)))
				
			model.add(Flatten())
			
			for l in range(dense_layer):
				model.add(Dense(layer_size))
				model.add(Activation('relu'))
				model.add(Dropout(0.2))
				
			model.add(Dense(1))
			model.add(Activation('sigmoid'))
			
			tb = TensorBoard(log_dir="logs/{}".format(NAME))
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
			
			model.fit(X, y, batch_size=32, epochs=num_epoch, validation_split=0.1, 
						callbacks=[tb], verbose=2.0)
			
			print("Model summary:")
			print(model.summary())
			
			if not os.path.exists(r'./models'):
				os.mkdir(r'./models')
			model.save(r'./models/' + str(NAME) + '.model')
			
			print("\nModel saved!")

print "\nDONE!"
