import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F

train = datasets.MNIST(root='./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST(root='./data', train=False, download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))


trainset = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

print "training = {}, testing = {} ".format(len(trainset.dataset), len(testset.dataset))

for epoch in range(3):
	print "ep = ", epoch
	for data in trainset:
		print "data length = ",len(data)
		X, y = data
		print "y = ", y
		break
        
