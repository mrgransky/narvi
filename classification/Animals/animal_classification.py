import os, cv2, time, sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import style

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	print "#" * 50
	print "\t\tRunning on GPU ..."
	print "#" * 50
else:
	device = torch.device("cpu")
	print "#" * 50
	print "\t\tRunning on CPU ..."
	print "#" * 50

MODEL_NAME = "model_{}".format(int(time.time()))

#REBUILD_DATA = True
REBUILD_DATA = False

class DogsVSCats():
    IMG_SIZE = 50
    
    data_dir = "/home/xenial/Datasets/Animals/Train"
    
    CATS = os.path.join(data_dir,'Cat')
    DOGS = os.path.join(data_dir,'Dog')
    
    print "cat dir = {}".format(CATS)
    print "dog dir = {}".format(DOGS)

    LABELS = {CATS: 0, DOGS: 1}
    training_data = []

    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        #print "img: {}".format(img.shape)
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
                        if label == self.CATS:
                            self.catcount += 1
                        elif label == self.DOGS:
                            self.dogcount += 1
                    except Exception as e:
                        pass
                        #print(label, f, str(e))

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print('Cats:',dogsvcats.catcount)
        print('Dogs:',dogsvcats.dogcount)

if REBUILD_DATA:
	print "REBUILD_DATA ..."
	dogsvcats = DogsVSCats()
	dogsvcats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)
print "Training data:\t len: {} , shape: {}".format(len(training_data), training_data.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 2) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)

net = Net().to(device)
print "\nModel summary:\n", net



#lossFCN = nn.CrossEntropyLoss()
#lossFCN = nn.BCELoss(size_average=True)
lossFCN = nn.MSELoss()

#optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
#optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.5, weight_decay=0.5)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# convert to tensor:
X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

print "\nRAW:\tX: {} \t y: {}".format(X.shape, y.shape)

VAL_PCT = 0.1  # lets reserve 10% of our data for validation
val_size = int(len(X)*VAL_PCT)
#print "validation sz = ", val_size

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print "\ntrain_X = {} \t test_X = {}" .format(train_X.shape, test_X.shape)

def fwd_pass(X, y, train=False):
	if train:
		net.zero_grad()

	outputs = net(X)
	matches  = [torch.argmax(i)==torch.argmax(j) for i, j in zip(outputs, y)]
	
	acc = float(matches.count(True))/len(matches)
	loss = lossFCN(outputs, y)
	
	if train:
		loss.backward()
		optimizer.step()
	return acc, loss

def train():
	BATCH_SIZE = 100
	EPOCHS = 8
	if not os.path.exists(r'./models'):
				os.mkdir(r'./models')
				
	with open(r'./models/' + str(MODEL_NAME) + ".txt", "a") as f:
		f.write("ts,loss,acc,val_loss,val_acc\n")
		for epoch in range(EPOCHS):
			print "Epoch {}/{} ".format(epoch + 1, EPOCHS)
			print "-"*20
			for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
				#print "\n\nX_train: {}\ty_train: {}".format(train_X.shape, train_y.shape)
				batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
				batch_y = train_y[i:i+BATCH_SIZE]
			
				batch_X, batch_y = batch_X.to(device), batch_y.to(device)
				"""
				print "\n{} < batch < {}:\tbatch_X: {} , batch_y: {}".format(i, i + BATCH_SIZE, 
																									batch_X.shape,
																									batch_y.shape)			
				"""
				acc, loss = fwd_pass(batch_X, batch_y, train=True)
				if i % 50 == 0:
					val_acc, val_loss = test(size=100)
					print "\nEpoch {}: loss = {}, acc: {}, val_loss = {}, val_acc: {}".format(epoch+1,
																					 loss, acc, val_loss, val_acc)
					f.write("%s,%s,%s,%s,%s\n" % (int(time.time()), 
																float(loss), acc, 
																float(val_loss), val_acc))
				
				print "-" * 90
			
def test(size=32):
	#print "\n\ntest_X: {}\t test_y: {} ".format(test_X.shape, test_y.shape)
	rand_start = np.random.randint(len(test_X) - size)
	X, y = test_X[rand_start:rand_start+size], test_y[rand_start:rand_start+size]
	with torch.no_grad():
		val_acc, val_loss = fwd_pass(X.view(-1,1,50,50).to(device), y.to(device))
	return val_acc, val_loss

print "#" * 40
print "\t\tTRAINING"
print "#" * 40
train()
