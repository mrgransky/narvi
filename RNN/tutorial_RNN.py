import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.compat.v1.keras.layers import Dense, Flatten, Dropout, LSTM#, CuDNNLSTM
from collections import deque
from sklearn import preprocessing
from tensorflow.keras.callbacks import TensorBoard
from time import time


# mnist is a dataset of 28x28 images of handwritten digits and their labels
mnist = tf.keras.datasets.mnist

# unpacks images to x_train/x_test and labels to y_train/y_test
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

#print(x_train[0])
#plt.imshow(x_train[5], cmap = plt.cm.binary)
#plt.show()

model = Sequential()
model.add(Flatten(input_shape=(x_train.shape[1:])))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


model.save('epic_num_reader.model')

new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict([x_test])
print(predictions)

print(np.argmax(predictions[0]))
plt.imshow(x_test[0])
plt.show()


"""
x_train = x_train/255.0
x_test = x_test/255.0
model = Sequential()

# CPU:
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))

# GPU:
#model.add(CuDNNLSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))

model.add(Dropout(0.2))

# CPU:
model.add(LSTM(128, activation='relu'))

# GPU:
#model.add(CuDNNLSTM(128))

model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="/home/xenial/logs/{}".format(time()))
model.fit(x_train, y_train, epochs = 3, validation_data=(x_test, y_test))

"""
