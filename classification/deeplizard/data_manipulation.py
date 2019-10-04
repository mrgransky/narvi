import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

import time
import numpy as np
import sklearn

train_path 	= '/home/xenial/WS_Farid/keras_ws/classification/deeplizard/train'
test_path 	= '/home/xenial/WS_Farid/keras_ws/classification/deeplizard/test'
valid_path 	= '/home/xenial/WS_Farid/keras_ws/classification/deeplizard/valid'





IMG_SIZE = 224

train_batches 	= ImageDataGenerator().flow_from_directory(train_path, target_size=(IMG_SIZE,IMG_SIZE), classes=['dog','cat'], batch_size=10)
test_batches 	= ImageDataGenerator().flow_from_directory(test_path, target_size=(IMG_SIZE,IMG_SIZE), classes=['dog','cat'], batch_size=4)
valid_batches 	= ImageDataGenerator().flow_from_directory(valid_path, target_size=(IMG_SIZE,IMG_SIZE), classes=['dog','cat'], batch_size=10)

def plots(ims, figsize=(12,6), rows=1,interp=False, titles=None):
	if type(ims[0]) is np.ndarray:
		ims = np.array(ims).astype(np.uint8)
		if (ims.shape[-1] != 3):
			ims = ims.transpose((0,2,3,1))
	f = plt.figure(figsize=figsize)
	cols = len(ims)
	for i in range(len(ims)):
		sp = f.add_subplot(rows, cols, i+1)
		sp.axis('off')
		if titles is not None:
			sp.set_title(titles[i],fontsize=16)
		plt.imshow(ims[i], interpolation=None if interp else 'none')
	plt.show()
		
#imgs, labels = next(train_batches)
#plots(imgs, titles=labels)
 
# build and train CNN:
model = Sequential()

# input layer:
model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Activation('relu'))
model.add(Flatten())

#model.add(Dense(64))
#model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)

print("Model summary:")
print(model.summary())

