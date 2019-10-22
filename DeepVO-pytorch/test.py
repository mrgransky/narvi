import glob, os, time, torch, sys
from tqdm import tqdm 
from params import par
from model import DeepVO
import numpy as np
from PIL import Image
from data_helper import generate_data, LoadMyDataset
from torch.utils.data import DataLoader
from helper import eulerAnglesToRotationMatrix
sys.dont_write_bytecode = True


if torch.cuda.is_available():
	#device = torch.device("cuda:0")
	#use_cuda = True
	print "#" * 50
	print "\t\tRunning on GPU ..."
	print "#" * 50
else:
	#device = torch.device("cpu")
	#use_cuda = False
	print "#" * 50
	print "\t\tRunning on CPU ..."
	print "#" * 50

if __name__ == '__main__':
	sequences = ['04', '05', '07', '09', '10']
	# Load model
	M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
	print "Loading model:\t{}".format(par.load_model_path)
	#M_deepvo.to(device)
	try:
		if par.use_cuda:
			M_deepvo = M_deepvo.cuda()
			M_deepvo.load_state_dict(torch.load(par.load_model_path))
		else:
			M_deepvo.load_state_dict(torch.load(par.load_model_path, map_location={'cuda:0': 'cpu'}))
	except Exception as e:
		print str(e)
		sys.exit()
	
	print "\n\nModel Summary:\n{}".format(M_deepvo)
	# Data
	n_workers = 1
	seq_len = int((par.seq_len[0]+par.seq_len[1])/2)
	overlap = seq_len - 1
	
	print "seq_len = {}\toverlap = {}".format(seq_len, overlap)
	batch_size = par.batch_size
	
	fd = open('test_dump.txt', 'w')
	fd.write('\n'+'='*50 + '\n')

	for seq in sequences:
		print "seq: {}".format(seq)
		df = generate_data([seq], seq_len_range=[seq_len, seq_len], overlap=overlap, ts=1, shuffle=False, sort=False)

		df = df.loc[df.seq_len == seq_len]  # drop last
		
		dataset = LoadMyDataset(df, par.resize_mode, (par.img_w, par.img_h), 
										par.img_means, par.img_stds, par.minus_point_5)
		
		df.to_csv('test_df.csv')
		
		dataloader = DataLoader(dataset, batch_size=batch_size, 
										shuffle=False, num_workers=n_workers)
		try:
			gt_pose = np.load('{}{}.npy'.format(par.pose_dir, seq))  # (n_images, 6)
		except Exception as e:
			print str(e)
			sys.exit()


		# Predict
		M_deepvo.eval()
		has_predict = False
		answer = [[0.0]*6, ]
		st_t = time.time()
		n_batch = len(dataloader)
		
		#for i, batch in enumerate(dataloader):
		for i, batch in tqdm(enumerate(dataloader)):
			#print('{} / {}'.format(i, n_batch), end='\r', flush=True)
			print "\n{}/{}".format(i, n_batch)
			_, x, y = batch
			if par.use_cuda:
				x = x.cuda()
				y = y.cuda()
			batch_predict_pose = M_deepvo.forward(x)

			# Record answer
			fd.write('Batch: {}\n'.format(i))
			for sq, predict_pose_seq in enumerate(batch_predict_pose):
				for pose_idx, pose in enumerate(predict_pose_seq):
					fd.write(' {} {} {}\n'.format(sq, pose_idx, pose))

			batch_predict_pose = batch_predict_pose.data.cpu().numpy()
			if i == 0:
				for pose in batch_predict_pose[0]:
					# use all predicted pose in the first prediction
					for i in range(len(pose)):
						# Convert predicted relative pose to absolute pose by adding last pose
						pose[i] += answer[-1][i]
					answer.append(pose.tolist())
				batch_predict_pose = batch_predict_pose[1:]

			# transform from relative to absolute 
			
			for predict_pose_seq in batch_predict_pose:
				# predict_pose_seq[1:] = predict_pose_seq[1:] + predict_pose_seq[0:-1]
				ang = eulerAnglesToRotationMatrix([0, answer[-1][0], 0]) #eulerAnglesToRotationMatrix([answer[-1][1], answer[-1][0], answer[-1][2]])
				location = ang.dot(predict_pose_seq[-1][3:])
				predict_pose_seq[-1][3:] = location[:]

			# use only last predicted pose in the following prediction
				last_pose = predict_pose_seq[-1]
				for i in range(len(last_pose)):
					last_pose[i] += answer[-1][i]
				# normalize angle to -Pi...Pi over y axis
				last_pose[0] = (last_pose[0] + np.pi) % (2 * np.pi) - np.pi
				answer.append(last_pose.tolist())

		print('len(answer): ', len(answer))
		print('expect len: ', len(glob.glob('{}{}/*.png'.format(par.image_dir, seq))))
		print('Predict use {} sec'.format(time.time() - st_t))

		# Save answer
		with open('{}/out_{}.txt'.format(par.save_dir, seq), 'w') as f:
			for pose in answer:
				if type(pose) == list:
					f.write(', '.join([str(p) for p in pose]))
				else:
					f.write(str(pose))
				f.write('\n')

		# Calculate loss
		#gt_pose = np.load('{}{}.npy'.format(par.pose_dir, seq))  # (n_images, 6)
		try:
			gt_pose = np.load('{}{}.npy'.format(par.pose_dir, seq))  # (n_images, 6)
		except Exception as e:
			print str(e)
			sys.exit()

		loss = 0
		for t in range(len(gt_pose)):
			angle_loss = np.sum((answer[t][:3] - gt_pose[t,:3]) ** 2)
			translation_loss = np.sum((answer[t][3:] - gt_pose[t,3:6]) ** 2)
			loss = (100 * angle_loss + translation_loss)
		loss /= len(gt_pose)
		print('Loss = ', loss)
		print('='*50)
