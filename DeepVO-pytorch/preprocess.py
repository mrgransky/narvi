import os, glob, math, torch, time
import numpy as np
from helper import R_to_angle
from params import par
from torchvision import transforms
from PIL import Image

def clean_unused_images():
	seq_frame = {	'00': ['000', 		'004540'],
						'01': ['000', 		'001100'],
						'02': ['000', 		'004660'],
						'03': ['000', 		'000800'],
						'04': ['000', 		'000270'],
						'05': ['000', 		'002760'],
						'06': ['000', 		'001100'],
						'07': ['000', 		'001100'],
						'08': ['001100', 	'005170'],
						'09': ['000', 		'001590'],
						'10': ['000', 		'001200']
					}
	print "seq_frame items :{}".format(seq_frame.items())
	for seq, img_ids in seq_frame.items():
		dir_path = '{}{}/'.format(par.image_dir, seq)
		if not os.path.exists(dir_path):
			print "#"*80
			print "path: {} does not exist!!".format(dir_path)
			print "#"*80
			continue
			
		start, end = img_ids
		start, end = int(start), int(end)
		
		print "Cleaning Sequence: {}\tstarting: {}\tends: {}".format(seq, start, end)
		
		for idx in range(0, start):
			img_name = '{:010d}.png'.format(idx)
			img_path = '{}{}/{}'.format(par.image_dir, seq, img_name)
			if os.path.isfile(img_path):
				print "removing:\t{}".format(img_path)
				os.remove(img_path)
		
		for idx in range(end+1, 10000):
			img_name = '{:010d}.png'.format(idx)
			img_path = '{}{}/{}'.format(par.image_dir, seq, img_name)
			if os.path.isfile(img_path):
				print "removing:\t{}".format(img_path)
				os.remove(img_path)
				
	print "#" * 20 
	print "cleaning DONE!"
	print "#" * 20

# transform poseGT [R|t] to [theta_x, theta_y, theta_z, x, y, z]
# save as .npy file
def create_pose_data():
	info = { '00': [0, 		4540],
				'01': [0, 		1100], 
				'02': [0, 		4660], 
				'03': [0, 		800], 
				'04': [0, 		270], 
				'05': [0, 		2760], 
				'06': [0, 		1100], 
				'07': [0, 		1100], 
				'08': [1100, 	5170], 
				'09': [0, 		1590], 
				'10': [0, 		1200]
				}
	print 'info_key {}'.format(info.keys())
		
	start_t = time.time()
	for seq in info.keys():
		fn = '{}{}.txt'.format(par.pose_dir, seq)
		print 'Transforming\t{}'.format(fn)
		with open(fn) as f:
			lines = [line.split('\n')[0] for line in f.readlines()] 
			poses = [ R_to_angle([float(value) for value in l.split(' ')]) for l in lines]
			poses = np.array(poses)
			#print "POSE = {}".format(poses)
			base_fn = os.path.splitext(fn)[0]
			np.save(base_fn + '.npy', poses)
			print('seq {}: shape = {}'.format(seq, poses.shape))
	print('elapsed time = {}'.format(time.time()-start_t))


def get_mean_std(image_path_list, minus_point_5=False):
	n_images = len(image_path_list)
	cnt_pixels = 0
	print "no images: {}".format(n_images)
	std_tensor, std_np, mean_tensor, mean_np = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
	
	to_tensor = transforms.ToTensor()
	image_sequence = []
	start_time_mean = time.time()
	for idx, img_path in enumerate(image_path_list):
		raw_IMG = Image.open(img_path)
		#print "FRAME {} / {}".format(idx, n_images)
		
		tensor_IMG = to_tensor(raw_IMG)
		if minus_point_5:
			tensor_IMG = tensor_IMG - 0.5
		
		np_IMG = np.array(raw_IMG)
		#print "shape:\tnp_img {}\ttensor_img: {}".format(np_IMG.shape,tensor_IMG.shape)
		
		np_IMG = np.rollaxis(np_IMG, 2) #(HxWxC) --> (CxHxW)
		#print "shape:\tROLLED_np_img {}".format(np_IMG.shape)
		cnt_pixels += np_IMG.shape[1]*np_IMG.shape[2]
		for c in range(3):
			mean_tensor[c] += float(torch.sum(tensor_IMG[c]))
			mean_np[c] 		+= float(np.sum(np_IMG[c]))
			
	mean_tensor =  [v / cnt_pixels for v in mean_tensor]
	mean_np 		=  [v / cnt_pixels for v in mean_np]
	print "mean:\ntensor = {}\nnp = {}".format(mean_tensor, mean_np)

	print ("\nTaken time:\t {} sec.".format((time.time() - start_time_mean)))


	start_time_std = time.time()
	for idx, img_path in enumerate(image_path_list):
		#print "FRAME: {} / {}".format(idx, n_images)
		raw_IMG = Image.open(img_path)
		tensor_IMG = to_tensor(raw_IMG)
		if minus_point_5:
			tensor_IMG = tensor_IMG - 0.5
		np_IMG = np.array(raw_IMG)
		np_IMG = np.rollaxis(np_IMG, 2, 0)
		for c in range(3):
			tmp 					= (tensor_IMG[c] - mean_tensor[c])**2
			std_tensor[c] 		+= float(torch.sum(tmp))
			
			tmp 					= (np_IMG[c] - mean_np[c])**2
			std_np[c] 			+= float(np.sum(tmp))
			
	std_tensor 	= [math.sqrt(v / cnt_pixels) for v in std_tensor]
	std_np 		= [math.sqrt(v / cnt_pixels) for v in std_np]
	
	print "std:\ntensor = {}\nnp = {}".format(std_tensor, std_np)
	print "\nTaken time:\t {} sec".format((time.time() - start_time_std))
	
if __name__ == '__main__':
	clean_unused_images()
	create_pose_data()
	sequences = ['00', '02', '08', '09', '06', '04', '10']
	#sequences = ['04']
	
	image_path_list = []
	for sq in sequences:
		try:
			print "Loading images:\t{}{}".format(par.image_dir, sq)
			image_path_list += glob.glob('{}{}/*.png'.format(par.image_dir, sq))
			image_path_list += glob.glob('{}{}/image_0/*.png'.format(par.image_dir, sq))
		except Exception as e:
			print str(e)
			sys.exit()
	get_mean_std(image_path_list, minus_point_5=True)
