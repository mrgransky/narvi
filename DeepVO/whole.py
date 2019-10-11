import glob
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from torch.autograd import Function
from torch.autograd import Variable
from torchvision import models
import math

#Function to get resized image from a provided path
def get_image(path,img_size= (1280,384)):
    img = cv2.imread(path)
    img = cv2.resize(img, img_size, cv2.INTER_LINEAR)
    return img
    
#Helper function to laod images from a given directory and forming batches
def load_images(img_dir, img_size):
    print ("images ", img_dir)
    images= []
    images_set =[]
    for img in glob.glob(img_dir+'/*'):
        images.append(get_image(img,img_size))
    for i in range(len(images)-1):
        img1 = images[i]
        img2 = images[i+1]
        img = np.concatenate([img1, img2],axis = -1)
        images_set.append(img)
    print("images count : ",len(images_set))
    images_set = np.reshape(images_set, (-1, 6, 384, 1280))
    return images_set

#Helper functions for pose preprocessing 
def isRotationMatrix(R):
    """ Checks if a matrix is a valid rotation matrix
        referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    """ calculates rotation matrix to euler angles
        referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles
    """
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def getMatrices(all_poses):
    all_matrices = []
    for i in range(len(all_poses)):
        #print("I: ",i)
        j = all_poses[i]
        #print("J:   ",j)
        p = np.array([j[3], j[7], j[11]])
        #print("P:   ", p)
        R = np.array([[j[0],j[1],j[2]],
                [j[4],j[5],j[6]],
                [j[8],j[9],j[10]]])
        #print("R:   ", R)
        angles = rotationMatrixToEulerAngles(R)
        #print("Angles: ",angles)
        matrix = np.concatenate((p,angles))
        #print("MATRIX: ", matrix)
        all_matrices.append(matrix)
    return all_matrices
    
    
#Helper function to get poses form a given location 
def load_poses(pose_file):
    print ("pose ",pose_file)
    poses = []
    poses_set = []
    with open(pose_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pose = np.fromstring(line, dtype=float, sep=' ')
            poses.append(pose)
    poses = getMatrices(poses)
    for i in range(len(poses)-1):
        pose1 = poses[i]
        pose2 = poses[i+1]
        finalpose = pose2-pose1
        poses_set.append(finalpose)
    print("poses count: ",len(poses_set))
    return poses_set          

#Primary dataloader function to get both images and poses
def VODataLoader(datapath,img_size= (1280,384), test=False):
    print (datapath)
    poses_path = os.path.join(datapath,'poses')
    img_path = os.path.join(datapath,'sequences')
    if test:
        sequences = ['03']  #Kindly use this sequence only for testing as this has mininum number of images
    else:
        #Uncomment below and comment the next to next line to work with larger data 
        #sequences= ['01','03','06']
        sequences = ['03']  
        
    images_set = []
    odometry_set = []
    
    for sequence in sequences:
        images_set.append(torch.FloatTensor(load_images(os.path.join(img_path,sequence,'image_0'),img_size)))
        odometry_set.append(torch.FloatTensor(load_poses(os.path.join(poses_path,sequence+'.txt'))))
    
    return images_set, odometry_set
    
#dataload
X,y = VODataLoader("/home/xenial/Datasets/KITTI")

print("Details of X :")
# print(type(X)) 
# print(type(X[0]))
# print(len(X)) 
# print(len(X[0])) 
print(X[0].size())
print("Details of y :")
# print(type(y))
# print(type(y[0]))
# print(len(y))
# print(len(y[0]))
print(y[0].size())


#Converting lists containing tensors to tensors as per the batchsize (10)
X=torch.stack(X).view(-1,10,6, 384, 1280)
y=torch.stack(y).view(-1,10,6)
print("Details of X :")
print(X.size())
print("Details of y :")
print(y.size())

#Helper function to display image
def imshow(img):
    plt.figure
    plt.imshow(img, 'gray')
    plt.show()
    
#Defining neural network as per RP by Sen Wang
class DeepVONet(nn.Module):
    def __init__(self):
        super(DeepVONet, self).__init__()

        self.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)) #6 64
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d (64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d (128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3_1 = nn.Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d (256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d (512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu5 = nn.ReLU(inplace=True)
        self.conv5_1 = nn.Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d (512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.lstm1 = nn.LSTMCell(20*6*1024, 100)
        self.lstm2 = nn.LSTMCell(100, 100)
        self.fc = nn.Linear(in_features=100, out_features=6)

        self.reset_hidden_states()

    def reset_hidden_states(self, size=10, zero=True):
        if zero == True:
            self.hx1 = Variable(torch.zeros(size, 100))
            self.cx1 = Variable(torch.zeros(size, 100))
            self.hx2 = Variable(torch.zeros(size, 100))
            self.cx2 = Variable(torch.zeros(size, 100))
        else:
            self.hx1 = Variable(self.hx1.data)
            self.cx1 = Variable(self.cx1.data)
            self.hx2 = Variable(self.hx2.data)
            self.cx2 = Variable(self.cx2.data)

        if next(self.parameters()).is_cuda == True:
            self.hx1 = self.hx1.cuda()
            self.cx1 = self.cx1.cuda()
            self.hx2 = self.hx2.cuda()
            self.cx2 = self.cx2.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv6(x)
        #print(x.size())
        x = x.view(x.size(0), 20 * 6 * 1024)
        #print(x.size())
        self.hx1, self.cx1 = self.lstm1(x, (self.hx1, self.cx1))
        x = self.hx1
        self.hx2, self.cx2 = self.lstm2(x, (self.hx2, self.cx2))
        x = self.hx2
        #print(x.size())
        x = self.fc(x)
        return x
        
        
# Training Function
def training_model(model, train_num, X, y, epoch_num=25):
    start_time = time.time()
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        running_loss = 0.0
        print("Epoch : ", epoch+1)
        for i in range(train_num):
            print("Train num :", i+1)
            inputs = X[i]
            #print(len(inputs))
            labels = y[i]
            #print(len(labels))
            model.zero_grad()
            model.reset_hidden_states()
            #optimizer.zero_grad()
            outputs = model(inputs)
            if(epoch == (epoch_num-1) and (i > train_num-5)):
                print("Outputs ", outputs)
                print("Labels ", labels)
            #print(outputs)
            #print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('Epoch : %d Loss: %.3f' %(epoch+1, running_loss/train_num))


    print('Finished Training')
    print ("Time taken in Training {0}".format((time.time() - start_time)))

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
            diff+= get_mse_diff(out,lab)
            print "\ndiff = ", diff
            print "\n\n------------------------------------------------\n\n"
    print "Loss : ", diff/(batch_size*10) ," %"
    print "Accuracy : ",(1 -diff/(batch_size*10))*100, " %"
    
def get_mse_diff(x,y):
    diff= 0
    for i in range(6):
        diff += (x[i]-y[i])*(x[i]-y[i])
    return diff/6

#Creating model and defining loss and optimizer to be used 
model = DeepVONet()
print(model)

import torch.optim as optim

#criterion = nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5, weight_decay=0.5)

#Uncomment lines below to see model paramters 
for parameter in model.parameters():
    print(len(parameter))
    

training_model(model,10,X,y,2)

#Save the model
torch.save(model.state_dict(), 'DeepVO.pt')
#Load model
model_loaded = torch.load('DeepVO.pt')

X_test,y_test = VODataLoader("/home/xenial/Datasets/KITTI", test=True)
X_test=torch.stack(X_test).view(-1,10,6, 384, 1280)
y_test=torch.stack(y_test).view(-1,10,6)
print(X_test.size())
print(y_test.size())


#Getting predictions from the model 
test_batch_size = 2 

y_output = testing_model(model, test_batch_size, X)

print "y_output size = " , y_output.size()
print "y_output shape = " , y_output.shape[0]

torch.save(y_output,"y_output.pt")
print "output saved successfully!"

#getting accuracy
get_accuracy(y_output, y_test, test_batch_size)
