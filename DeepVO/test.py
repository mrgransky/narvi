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
			out = outputs[j].numpy()
			lab = labels[j].numpy()
			diff+=get_mse_diff(out,lab)
	#print("Loss : ",diff/(batch_size*10),"%")
	print("Accuracy : ",(1 -diff/(batch_size*10))*100,"%")
    
def get_mse_diff(x,y):
	diff= 0
	for i in range(6):
		diff += (x[i]-y[i])*(x[i]-y[i])
	return diff/6

if len(sys.argv) != 2:
	print "\nSYNTAX: \n\npython " + sys.argv[0] + " [PATH/2/model.pt]\n"
	sys.exit()

#Testing functions, it is predicting the output for test sequence as per the model
def testing_model (model, test_num, X):
	start_time = time.time()
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

path = sys.argv[1]

print "\nLoading model..."
model = DeepVONet()
model.load_state_dict(torch.load(path))
print "\nEvaluating model :"
print model.eval()


dataloader = DataPreparation("/home/xenial/Datasets/KITTI")
X_test,y_test = dataloader.VODataLoader(test=True)

X_test = torch.stack(X_test).view(-1, 10, 6, 384, 1280)
y_test = torch.stack(y_test).view(-1, 10, 6)

print "\n\nX_test.size = " , X_test.size()
print "\n\ny_test.size = " , y_test.size()

#Getting predictions from the model 
test_batch_size = 2
y_output = testing_model(model, test_batch_size, X)

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
