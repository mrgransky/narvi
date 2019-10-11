import numpy as np
import os, time, sys, math, cv2, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
from torchvision import models
import torch.optim as optim
import matplotlib.pyplot as plt
from deep_vo_net import DeepVONet
from generate_Xy import DataPreparation

#Helper functions to get accuracy
def get_accuracy(outputs, labels, batch_size):
	diff =0
	for i in range(batch_size):
		for j in range(10):
			print "\n\n#################################################n\n"
			print "\n\n(i,j) = \t ", i , " , ", j
			out = outputs[j].detach().numpy()
			print "\nout[", j,"] = ", out
			lab = labels[j].detach().numpy()
			print "\nlab[", j,"] = ", lab
			diff+=get_mse_diff(out,lab)
			print "\ndiff = ", diff
			print "\n\n------------------------------------------------\n\n"
	print "Loss : ",diff/(batch_size*10)," %"
	print "Accuracy : ",(1 -diff/(batch_size*10))*100," %"
    
def get_mse_diff(x,y):
	diff= 0
	for i in range(6):
		diff += (x[i]-y[i])*(x[i]-y[i])
	return diff/6

if len(sys.argv) != 3:
	print "\nSYNTAX: \n\npython " + sys.argv[0] + " [PATH/2/Xy folder] [PATH/2/model.pt]\n"
	sys.exit()

def load_pytorch_model(path):
	print "\nLoading model..."
	model = DeepVONet()
	model.load_state_dict(torch.load(path))
	return model
	
def load_Xy_train(path):
	print "\nLoading training data X,y from: ", str(path) + 'train/'
	X = torch.load(path + 'train/' + 'X.pt')
	y = torch.load(path + 'train/' + 'y.pt')
	return X,y

X_train,y_train = load_Xy_train(sys.argv[1])
print "\n\nX & y TRAIN loaded successfully!"

def load_Xy_test(path):
	print "\nLoading testing data X,y from: ", str(path) + 'test/'
	X = torch.load(path + 'test/' + 'X.pt')
	y = torch.load(path + 'test/' + 'y.pt')
	return X,y

def testing_model (model, test_num, X):
	start_time = time.time()
	print "start_time:\t", start_time
	Y_output = []
	count = 0
	totcount = 0
	for i in range(test_num):
		# get the inputs
		inputs = X[i]
		outputs = model(inputs)
		Y_output.append(outputs)
	print ("Time taken in Testing {0}".format((time.time() - start_time)))
	return torch.stack(Y_output)

X_test,y_test = load_Xy_test(sys.argv[1])
print "\n\nX & y TEST loaded successfully!"
sp
model = load_pytorch_model(sys.argv[2])
print "\nModel loaded successfully, Evaluating ...\n\n"

print model.eval()
	

X_test = torch.stack(X_test).view(-1, 10, 6, 384, 1280)
y_test = torch.stack(y_test).view(-1, 10, 6)

print "\n\nX_test.size = " , X_test.size()
print "\n\ny_test.size = " , y_test.size()

#Getting predictions from the model 
test_batch_size = 2

###### ERRRRRROOOORRRR: ######
y_output = testing_model(model, test_batch_size, X_train)

print "\ny_output.size = " , y_output.size()

print "\nSaving output ..."
now = datetime.now() # current date and time
date_time = now.strftime("%d%m%Y%H%M%S")
print "\nTime:", date_time
NAME = "out_DeepVO_{}".format(int(date_time))

if not os.path.exists(r'./outputs'):
	os.mkdir(r'./outputs')

torch.save(y_output, r'./outputs/' + str(NAME) + '.pt')

print("\nOutput saved successfully!!")

#getting accuracy
get_accuracy(y_output, y_test, test_batch_size)
