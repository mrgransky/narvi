import os, multiprocessing

class Parameters():
	def __init__(self):
		self.n_processors = multiprocessing.cpu_count()
		
		# local machine:
		self.data_dir =  '/home/xenial/Datasets/KITTI/'
		self.image_dir = self.data_dir + 'images/'
		self.pose_dir = self.data_dir + 'pose_GT/'
		
		"""
		# server machine:
		self.data_dir 	=  '/home/alijani/Datasets/kitti_color/'
		self.image_dir = self.data_dir + 'images/'
		self.pose_dir 	= self.data_dir + 'pose_GT/'
		"""
		
		self.train_seq = ['00', '01', '02', '05', '08', '09']
		self.test_seq 	= ['04', '06', '07', '10']
		
		#self.train_seq = ['04']
		#self.test_seq 	= ['04']
		
		self.partition = None  # partition videos in 'train_seq' to train / valid dataset  #0.8
		
		# Data Preprocessing
		self.resize_mode = 'rescale'  # choice: 'crop' 'rescale' None
		
		self.img_w, self.img_h = 608, 184
		
		self.img_means =  (0.19007764876619865, 0.15170388157131237, 0.10659445665650864)
		self.img_stds	=  (0.2610784009469139, 0.25729316928935814, 0.25163823815039915)
		self.minus_point_5 = True

		self.seq_len = (5,7)# in training LSTM to randomly segment seqs to subsequences 
		self.ts = 3

		# Data info path
		DeepVO_MISC_PATH = '/home/xenial/WS_Farid/DeepVO_misc/'
		
		self.train_df_path = '{}datainfo/train_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(DeepVO_MISC_PATH, ''.join(self.train_seq), ''.join(self.test_seq), 
																															self.partition, self.seq_len[0], self.seq_len[1], self.ts)
		
		self.test_df_path  = '{}datainfo/valid_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(DeepVO_MISC_PATH, ''.join(self.train_seq), ''.join(self.test_seq), 
																															self.partition, self.seq_len[0], self.seq_len[1], self.ts)
		
		# Model
		self.rnn_hidden_size = 1000
		self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
		self.rnn_dropout_out = 0.5
		self.rnn_dropout_between = 0   # 0: no dropout
		self.clip = None
		self.batch_norm = True
		
		# Training
		self.epochs = 10 #250
		self.batch_size = 8
		self.pin_mem = True
		self.optim = {'opt': 'Adagrad', 'lr': 5e-4}
					# Choice:
					# {'opt': 'Adagrad', 'lr': 0.001}
					# {'opt': 'Adam'}
					# {'opt': 'Cosine', 'T': 100 , 'lr': 0.001}
		
		# Pretrain, Resume training
		self.pretrained_flownet = None
								# Choice:
								# None
								# './pretrained/flownets_bn_EPE2.459.pth.tar'  
								# './pretrained/flownets_EPE1.951.pth.tar'
								
		self.model_available = True  # resume training
		self.resume_t_or_v = '.train'
		
		
		self.load_model_path 		= '{}models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model{}'.format(DeepVO_MISC_PATH, ''.join(self.train_seq), ''.join(self.test_seq), 
																														self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], 
																														self.batch_size, self.rnn_hidden_size, 
																														'_'.join([k+str(v) for k, v in self.optim.items()]), 
																														self.resume_t_or_v)
		
		self.load_optimizer_path 	= '{}models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer{}'.format(DeepVO_MISC_PATH, ''.join(self.train_seq), ''.join(self.test_seq), 
																															self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, 
																															self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), 
																															self.resume_t_or_v)

		self.record_path 				= '{}records/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.txt'.format(DeepVO_MISC_PATH, ''.join(self.train_seq), ''.join(self.test_seq), 
																															self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], 
																															self.batch_size, self.rnn_hidden_size, 
																															'_'.join([k+str(v) for k, v in self.optim.items()]))
		
		self.save_model_path 		= '{}models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(DeepVO_MISC_PATH, ''.join(self.train_seq), ''.join(self.test_seq), 
																															self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], 
																															self.batch_size, self.rnn_hidden_size, 
																															'_'.join([k+str(v) for k, v in self.optim.items()]))
		
		self.save_optimzer_path 	= '{}models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(DeepVO_MISC_PATH, ''.join(self.train_seq), ''.join(self.test_seq), 
																															self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], 
																															self.batch_size, self.rnn_hidden_size, 
																															'_'.join([k+str(v) for k, v in self.optim.items()]))
		
		self.save_dir = '{}result/'.format(DeepVO_MISC_PATH)
		
		
		if not os.path.isdir(os.path.dirname(self.record_path)):
			os.makedirs(os.path.dirname(self.record_path))
			
		if not os.path.isdir(os.path.dirname(self.save_model_path)):
			os.makedirs(os.path.dirname(self.save_model_path))
			
		if not os.path.isdir(os.path.dirname(self.save_optimzer_path)):
			os.makedirs(os.path.dirname(self.save_optimzer_path))
			
		if not os.path.isdir(os.path.dirname(self.train_df_path)):
			os.makedirs(os.path.dirname(self.train_df_path))
			
		if not os.path.isdir(os.path.dirname(self.save_dir)):
			os.makedirs(os.path.dirname(self.save_dir))
		



par = Parameters()
