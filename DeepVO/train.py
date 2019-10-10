import glob
import numpy as np
import os, time, sys, math, cv2, torch
#import cv2
#import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
from torchvision import models
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from deep_vo_net import DeepVONet
#from generate_Xy import DataPreparation

def load_Xy(path):
	print "\nLoading training data X,y from: ", str(path)
	X = torch.load(path + 'X.pt')
	y = torch.load(path + 'y.pt')
	return X,y

#dataloader = DataPreparation("/home/xenial/Datasets/KITTI")
#X, y = dataloader.VODataLoader()

if len(sys.argv) != 2:
	print "\nSYNTAX: \n\npython " + sys.argv[0] + " [PATH/2/Xy folder]\n"
	sys.exit()

X, y = load_Xy(sys.argv[1])
print "\n\nX & y loaded successfully!"


print "\n\nDetails of X :\n"
print "type :\t" 			, type(X) 
print "type X[0] :\t" 	, type(X[0])
print "shape X[0] :\t" 	, X[0].shape
print "len :\t" 			, len(X) 
print "len X[0] :\t" 	, len(X[0]) 
print "size X[0] :\t" 	, X[0].size()

print "\n\nDetails of y :\n"
print "type :\t" 			, type(y) 
print "type y[0] :\t" 	, type(y[0])
print "shape y[0] :\t" 	, y[0].shape
print "len :\t" 			, len(y) 
print "len y[0] :\t" 	, len(y[0]) 
print "size y[0] :\t" 	, y[0].size()
print "\n\n--------------------------------------\n\n"

#Converting lists containing tensors to tensors as per the batchsize (10)
X = torch.stack(X).view(-1, 10, 6, 384, 1280)
y = torch.stack(y).view(-1, 10, 6)

print "Details of X: " 	, X.size()
print "Details of y: "	, y.size()


#Helper function to display image
def imshow(img):
	plt.figure
	plt.imshow(img, 'gray')
	plt.show()

# Training Function
def training_model(model, train_num, X, y, epoch_num = 25):
	start_time = time.time()
	for epoch in range(epoch_num):
		running_loss = 0.0
		print "\nEpoch:\t", epoch + 1
		for i in range(train_num):
			print "\n\nTrain num:\t", i + 1
			inputs = X[i]
			#print "len(inp) =\t", len(inputs)
			labels = y[i]
			#print "len(lbl) =\t", len(labels)
			model.zero_grad()
			model.reset_hidden_states()
			#optimizer.zero_grad()
			outputs = model(inputs)
			if(epoch == (epoch_num - 1) and (i > train_num - 5)):
				print "\n\n --------------------------------------------------------- \n\n"
				print "epoch = ", epoch , "\t training num = ", i
				print "\n\nOutputs : ", outputs
				print "\n\nLabels : ", labels 
				print "\n\n --------------------------------------------------------- \n\n"
			#print "OUT : " 	, outputs
			#print "LABLE :" 	, labels
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			
			# print statistics
			running_loss += loss.item()
		print('Epoch : %d Loss: %.3f' %(epoch+1, running_loss/train_num))
	
	print('\n\n********** Finished Training **********\n\n')
	print ("Time taken in Training {0}".format((time.time() - start_time)))
	
#Creating model and defining loss and optimizer to be used 
model = DeepVONet()
print "\n\nModel:\n" , model

#criterion = nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5, weight_decay=0.5)

it = 0
# Observe model paramters 
for parameter in model.parameters():
	it = it + 1
	print "parameter[", it, "] = \t", len(parameter)


print "\n\n************** TRAINING ************** \n\n"
# train the model:
training_model(model, 10, X, y, 2)

print "\nSaving model..."
now = datetime.now() # current date and time
date_time = now.strftime("%d%m%Y%H%M%S")
print "\nTime:", date_time
NAME = "DeepVO_{}".format(int(date_time))

if not os.path.exists(r'./models'):
	os.mkdir(r'./models')

torch.save(model.state_dict(), r'./models/' + str(NAME) + '.pt')

print("\nModel saved successfully!!")
