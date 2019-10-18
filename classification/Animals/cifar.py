import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

btchSZ = 4
tf = transforms.Compose(	
					[transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tf)
trainset = torch.utils.data.DataLoader(train, batch_size=btchSZ, shuffle=True, num_workers=8)

test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tf)
testset = torch.utils.data.DataLoader(test, batch_size=btchSZ, shuffle=False, num_workers=8)

print "training = {}, testing = {} ".format(len(trainset.dataset), len(testset.dataset))

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainset)
images, labels = dataiter.next()

print("\nTRAINING DATA INFO:")
print "imgs = {}, lbl = {} ".format(images.shape, labels.shape)
print('-' * 10)

# show images
imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(btchSZ)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 	= nn.Conv2d(3, 6, 5)
        
        self.pool 	= nn.MaxPool2d(2, 2)
        
        self.conv2 	= nn.Conv2d(6, 16, 5)
        
        self.fc1 		= nn.Linear(16 * 5 * 5, 120)
        self.fc2 		= nn.Linear(120, 84)
        self.fc3 		= nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 16 * 5 * 5)
        
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        
        return x

net = Net()
print "\nModel Summary:\n" , net

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# Train the network
########################################################################

print "\n\n########### TRAINING ###########\n"
for epoch in range(2):
	running_loss = 0.0
	for batch_idx, data in enumerate(trainset):
		# get the inputs; data is a list of [inputs, labels]
		#print "data length = ",len(data)
		inputs, labels = data
		#print "ep = {}, i = {}, inp = {}, lbl = {}".format(epoch, i, inputs.shape, labels.shape)
		# zero the parameter gradients
		optimizer.zero_grad()
		# forward + backward + optimize
		#print "labels[", i, "] = ", labels
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		# print statistics
		running_loss += loss.item()
		#print "epoch = {} , batch_idx = {}".format(epoch + 1, batch_idx)
		"""
		# print every 2000 mini-batches
		if i % 2000 == 1999:
			print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0
		"""

print "\nFinished Training!"

print "\nsaving the model..."
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


print "\n\n########### TESTING ###########\n"
dataiter = iter(testset)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(btchSZ)))

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(btchSZ)))

correct 	= 0.0
total 	= 0.0

with torch.no_grad():
	#for data in testset:
	#for batch_idx, data in enumerate(testset):
	for batch_idx, data in enumerate(trainset):	
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		#print "batch_idx = {}, predicted = {}".format(batch_idx, predicted)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print "Accuracy over all test images:\t", (correct/total)*100, " %"

class_correct 	= list(0. for i in range(len(classes)))
class_total 	= list(0. for i in range(len(classes)))

with torch.no_grad():
	for data in testset:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs, 1)
		c = (predicted == labels).squeeze()
		for i in range(4):
			label = labels[i]
			class_correct[label] += c[i].item()
			class_total[label] += 1


for i in range(len(classes)):
	print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


# %%%%%%INVISIBLE_CODE_BLOCK%%%%%%
del dataiter
# %%%%%%INVISIBLE_CODE_BLOCK%%%%%%

