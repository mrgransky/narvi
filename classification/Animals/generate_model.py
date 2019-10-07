import tensorflow as tf
import numpy as np
import pandas as pd
import random
from datetime import datetime

import matplotlib.pyplot as plt
import cv2
import pickle
import sys, os

from tensorflow.keras.models import Sequential, load_model
from tensorflow.compat.v1.keras.layers import Dense, Flatten, Dropout, LSTM#, CuDNNLSTM
from collections import deque
from sklearn import preprocessing
from tensorflow.keras.callbacks import TensorBoard
from time import time
from tqdm import tqdm

if len(sys.argv) != 2:
	print "\nSYNTAX: " + sys.argv[0] + " [PATH/2/DATASET]"
	print "\n\nExample:\n\npython "+ sys.argv[0] + " /home/xenial/Datasets/Animals/Train\n"
	sys.exit()

DATADIR = sys.argv[1]

#CATEGORIES =["Dog", "Cat", "Panda"]
CATEGORIES =["Dog", "Cat"]
now = datetime.now() # current date and time
date_time = now.strftime("%d%m%Y%H%M%S")
print "\nTime:", date_time
features = "X_{}".format(int(date_time))
label = "y_{}".format(int(date_time))

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

if not os.path.exists(r'./inp_out'):
				os.mkdir(r'./inp_out')
				
pickle_out = open(r'./inp_out/' + str(features) + '.pickle', "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open(r'./inp_out/' + str(label) + '.pickle', "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print "Model generated successfully!"
