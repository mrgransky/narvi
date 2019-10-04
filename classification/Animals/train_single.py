import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

inp_epoch = 25;
inp_conv = 64;

NAME = "Cats-vs-dogs-cnn-{}x2-conv-{}-epoch-{}".format(inp_conv, inp_epoch, int(time.time()))
tb = TensorBoard(log_dir='logs/{}'.format(NAME))

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0 # [0 - 1]

model = Sequential()

# input layer:
model.add(Conv2D(inp_conv, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#
model.add(Conv2D(inp_conv, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

#model.add(Dense(64))
#model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=32, epochs=inp_epoch, validation_split=0.1, callbacks=[tb], verbose=2.0)
print("Model summary:")
print(model.summary())

