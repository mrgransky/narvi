import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import cv2
import pickle

from tensorflow.keras.models import Sequential, load_model
from tensorflow.compat.v1.keras.layers import Dense, Flatten, Dropout, LSTM#, CuDNNLSTM
from collections import deque
from sklearn import preprocessing
from tensorflow.keras.callbacks import TensorBoard
from time import time
from tqdm import tqdm

DATADIR = "/home/xenial/WS_Farid/keras_ws/classification/Animals/dataset"
#CATEGORIES =["Dog", "Cat", "Panda"]
CATEGORIES =["Dog", "Cat"]

IMG_SIZE = 50
training_data = []
def create_training_data():
	for category in CATEGORIES:
		path = os.path.join(DATADIR, category)
		class_num = CATEGORIES.index(category)
		for img in tqdm(os.listdir(path)):
			try:
				image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
				img_resized = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
				training_data.append([img_resized, class_num])
			except Exception as e:
				#pass
				print("GENERAL_EXCEPTION: ", e, os.path.join(path,img))
			#except OSError as e:
			#    print("OSErrroBad img most likely", e, os.path.join(path,img))
			#except Exception as e:
			#    print("general exception", e, os.path.join(path,img))


create_training_data()

print "\n\nnumber of images in total: " , len(training_data)

#plt.imshow(training_data[0][0], cmap='gray')
#plt.show()

random.shuffle(training_data)

X = []
y = []

# sample_	: images
# label_	: categories (dog or cat)
for sample_, label_ in training_data:
	X.append(sample_)
	y.append(label_)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print "Model generated successfully!"
