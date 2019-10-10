import numpy as np
import matplotlib.pyplot as plt
import os, time, sys, math, cv2, torch, glob
#import pickle
import dill
#import cPickle as cpkl
import hickle as hkl

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Function
from torch.autograd import Variable
from torchvision import models

from datetime import datetime
from sklearn.externals import joblib

class DataPreparation:
	def __init__(self, filePath):
		self.data_path = filePath

	def get_image(self, path, img_size= (1280,384)):
		img = cv2.imread(path)
		#print "\n\nimage orig shape =\t", img.shape
		img = cv2.resize(img, img_size, cv2.INTER_LINEAR) # bilinear interpolation (default)
		return img
	
	def load_images(self, img_dir, img_size):
		print "Loading frames from directory: ", img_dir
		images= []
		img_arr =[]
		for img in glob.glob(img_dir+'/*'):
			images.append(self.get_image(img, img_size))
		for i in range(len(images) - 1):
			img1 = images[i]
			img2 = images[i+1]
			#print "\nshape:\timg_1:\t", img1.shape , "\t img_2:\t", img2.shape
			img = np.concatenate([img1, img2], axis = -1)
			img_arr.append(img)
		print "no frames: ", len(img_arr)
		img_arr = np.reshape(img_arr, (-1, 6, 384, 1280))
		print "img_arr shape = ", img_arr.shape
		return img_arr
	
	def isRotationMatrix(self, R):
		Rt = np.transpose(R)
		shouldBeIdentity = np.dot(Rt, R)
		I = np.identity(3, dtype = R.dtype)
		n = np.linalg.norm(I - shouldBeIdentity)
		return n < 1e-6
	
	def rotationMatrixToEulerAngles(self, R):
		assert(self.isRotationMatrix(R))
		sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
		singular = sy < 1e-6
		if  not singular:
			x = math.atan2(R[2,1] , R[2,2])
			y = math.atan2(-R[2,0], sy)
			z = math.atan2(R[1,0], R[0,0])
		else:
			x = math.atan2(-R[1,2], R[1,1])
			y = math.atan2(-R[2,0], sy)
			z = 0
		return np.array([x, y, z])
	
	def getMatrices(self, all_poses):
		all_matrices = []
		for i in range(len(all_poses)):
			#print("I: ",i)
			j = all_poses[i]
			#print("J:   ",j)
			p = np.array([j[3], j[7], j[11]])
			#print("P:   ", p)
			R = np.array([[j[0],j[1],j[2]], [j[4],j[5],j[6]], [j[8],j[9],j[10]]])
			#print("R:   ", R)
			angles = self.rotationMatrixToEulerAngles(R)
			#print("Angles: ",angles)
			matrix = np.concatenate((p,angles))
			#print("MATRIX: ", matrix)
			all_matrices.append(matrix)
		return all_matrices
	
	def load_poses(self, pose_file):
		print "Loading GT from directory ", pose_file
		poses = []
		poses_set = []
		with open(pose_file, 'r') as f:
			lines = f.readlines()
			for line in lines:
				pose = np.fromstring(line, dtype=float, sep=' ')
				poses.append(pose)
		poses = self.getMatrices(poses)
		for i in range(len(poses) - 1):
			pose1 = poses[i]
			pose2 = poses[i+1]
			finalpose = pose2 - pose1
			poses_set.append(finalpose)
		print "no GT: ",len(poses_set)
		return poses_set

	def VODataLoader(self, img_size = (1280,384), test=False):
		print "\nDataset Path:\t", self.data_path
		poses_path 	= os.path.join(self.data_path, 'poses')
		img_path 	= os.path.join(self.data_path, 'sequences')
		if test:
			#sequences = ['03']
			sequences = ['03', '04', '05', '06', '07', '10']
		else:
			#sequences= ['00', '02', '08', '09']
			sequences = ['03']
		img_arr 	= []
		gt_arr 	= []
		for seq in sequences:
			path2frames = os.path.join(img_path, 	seq, 'image_0')
			path2GT		= os.path.join(poses_path, seq +'.txt')
		
			img_arr.append(torch.FloatTensor(self.load_images(path2frames, img_size)))
			gt_arr.append(torch.FloatTensor(self.load_poses(path2GT)))
			self.save_Xy(img_arr, gt_arr)
		return img_arr, gt_arr
	
	def save_Xy(self, X, y):
		if not os.path.exists(r'./inp_out'):
			os.mkdir(r'./inp_out')
		
		features = "X"
		label = "y"
		print "\nStarting dump ...\n"
		
		with open(r'./inp_out/' + str(features) + '.pkl', 'wb') as fX:
			dill.dump(X, fX)
		
		with open(r'./inp_out/' + str(label) + '.pkl', 'wb') as fy:
			dill.dump(y, fy)
		
		"""
		pickle_out = open(r'./inp_out/' + str(features) + '.pickle', "wb")
		pkl.dump(X, pickle_out, -1)
		pickle_out.close()

		pickle_out = open(r'./inp_out/' + str(label) + '.pickle', "wb")
		pkl.dump(y, pickle_out, -1)
		pickle_out.close()


		joblib_X = open(r'./inp_out/' + str(features) + '.sav', "wb")
		joblib.dump(X, joblib_X)
		joblib_X.close()
		
		joblib_y = open(r'./inp_out/' + str(label) + '.sav', "wb")
		joblib.dump(y, joblib_y)
		joblib_y.close()
		
		
		ff = open(r'./inp_out/' + str(features) + '.hkl', 'w')
		hkl.dump(X, ff)
		ff.close() 
		
		p = cpkl.Pickler(open("temp.p","wb")) 
		p.fast = True 
		p.dump(X) # d could be your dictionary or any file
		"""
		
		print "X \t & y \t saved successfully!"
		
if __name__ == '__main__':
	if len(sys.argv) != 2:
		print "\nSYNTAX: \n\npython " + sys.argv[0] + " [PATH/2/Dataset]\n"
		sys.exit()

	dataloader = DataPreparation(sys.argv[1])
	dataloader.VODataLoader()
