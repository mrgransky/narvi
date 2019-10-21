import torch, os, time, sys
from tqdm import tqdm 
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from params import par
from model import DeepVO
from data_helper import generate_data, SortedRandomBatchSampler, LoadMyDataset, get_partition_data_info

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	use_cuda = True
	print "#" * 50
	print "\t\tRunning on GPU ..."
	print "#" * 50
else:
	device = torch.device("cpu")
	use_cuda = False
	print "#" * 50
	print "\t\tRunning on CPU ..."
	print "#" * 50

# Write all hyperparameters to record_path
mode = 'a' if par.resume else 'w'
with open(par.record_path, mode) as f:
	f.write('\n'+'='*50 + '\n')
	f.write('\n'.join("%s: %s" % item for item in vars(par).items()))
	f.write('\n'+'='*50 + '\n')

# Prepare Data
print "#" * 50
print "\t\tPreparing Data"
print "#" * 50
if os.path.isfile(par.train_df_path) and os.path.isfile(par.test_df_path):
	print "\nData info already exists!"
	print "\nLoading data info from:\t{}".format(par.train_df_path)
	train_df = pd.read_pickle(par.train_df_path)
	valid_df = pd.read_pickle(par.test_df_path)
else:
	print "\nData info NOT found!"
	print "\nCreating new data info..."
	if par.partition != None:
		print "\nWith partition..."
		partition = par.partition
		train_df, valid_df = get_partition_data_info(partition, par.train_seq, par.seq_len, overlap=1, sample_times=par.ts, shuffle=True, sort=True)
	else:
		print "\nWith NO partition...\n"
		
		train_df = generate_data(par.train_seq, seq_len_range=par.seq_len, overlap=1, ts=par.ts)
		valid_df = generate_data(par.test_seq, seq_len_range=par.seq_len, overlap=1, ts=par.ts)
	# save the data info
	train_df.to_pickle(par.train_df_path)
	valid_df.to_pickle(par.test_df_path)

print "\n\nloading data (TRAINING)"

train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)

trainSET = LoadMyDataset(train_df, par.resize_mode, (par.img_w, par.img_h), 
									par.img_means, par.img_stds, par.minus_point_5)

train_dl = DataLoader(trainSET, batch_sampler=train_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

print "Training dataset, with {} samples".format(len(train_df.index))

print "\n\nloading data (TESTING)"
valid_sampler = SortedRandomBatchSampler(valid_df, par.batch_size, drop_last=True)

testSET = LoadMyDataset(valid_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)

valid_dl = DataLoader(testSET, batch_sampler=valid_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

print "Testing dataset with {} samples".format(len(valid_df.index))

# Model
M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm).to(device)
#print "\n\nModel Summary:\n{}".format(M_deepvo)

# Load FlowNet weights pretrained with FlyingChairs
# NOTE: the pretrained model assumes image rgb values in range [-0.5, 0.5]
if par.pretrained_flownet and not par.resume:
	if use_cuda:
		pretrained_w = torch.load(par.pretrained_flownet)
	else:
		pretrained_w = torch.load(par.pretrained_flownet_flownet, map_location='cpu')
		
	print "Loading FlowNet pretrained model ..."
	
	# Use only conv-layer-part of FlowNet as CNN for DeepVO
	model_dict 	= M_deepvo.state_dict()
	update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
	model_dict.update(update_dict)
	M_deepvo.load_state_dict(model_dict)

# Create optimizer
if par.optim['opt'] == 'Adam':
	optimizer 		= torch.optim.Adam(M_deepvo.parameters(), lr=1e-2, betas=(0.9, 0.999))
elif par.optim['opt'] == 'Adagrad':
	optimizer 		= torch.optim.Adagrad(M_deepvo.parameters(), lr=par.optim['lr'])
elif par.optim['opt'] == 'Cosine':
	optimizer 		= torch.optim.SGD(M_deepvo.parameters(), lr=par.optim['lr'])
	T_iter 			= par.optim['T']*len(train_dl)
	lr_scheduler 	= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_iter, eta_min=0, last_epoch=-1)

# Load trained DeepVO model and optimizer
if par.resume:
	print "\nTrying to Load ...\n\nmodel: {}\n\noptimizer:{}".format(par.pretrained_model,
																							par.pretrained_optimizer) 
	try:
		if use_cuda:
			M_deepvo.load_state_dict(torch.load(par.pretrained_model))
			optimizer.load_state_dict(torch.load(par.pretrained_optimizer))
		else:
			M_deepvo.load_state_dict(torch.load(par.pretrained_model, map_location='cpu'))
			optimizer.load_state_dict(torch.load(par.pretrained_optimizer, map_location='cpu'))
	except Exception as e:
		print str(e)
		sys.exit()

print "Recording loss in:\t{}".format(par.record_path)
min_loss_t, min_loss_v = 1e10, 1e10
M_deepvo.train()
for ep in range(par.epochs):
	st_t = time.time()
	# Train
	M_deepvo.train()
	loss_mean = 0
	t_loss_list = []
	print "\nlen train_df = {}".format(len(train_df))
	#for idx, t_x, t_y in train_dl:
	for idx, t_x, t_y in tqdm(train_dl):
		print "\nEpoch {}/{}".format(ep + 1, par.epochs)
		print "\nidx = {}\tt_x = {}\tty = {}".format(idx, t_x.shape, t_y.shape)
		print('-'*100)
		if use_cuda:
			t_x = t_x.cuda(non_blocking=par.pin_mem)
			t_y = t_y.cuda(non_blocking=par.pin_mem)

		ls = M_deepvo.step(t_x, t_y, optimizer).data.cpu().numpy()
		t_loss_list.append(float(ls))
		loss_mean += float(ls)
		if par.optim == 'Cosine':
			lr_scheduler.step()
	print 'Training time: {:.1f} [s]'.format(time.time()-st_t)
	loss_mean /= len(train_dl)

	print "\nlen valid_df = {}".format(len(valid_df))
	# Validation
	st_t = time.time()
	M_deepvo.eval()
	loss_mean_valid = 0
	v_loss_list = []
	#for _, v_x, v_y in valid_dl:
	for idx, v_x, v_y in tqdm(valid_dl):
		if use_cuda:
			v_x = v_x.cuda(non_blocking=par.pin_mem)
			v_y = v_y.cuda(non_blocking=par.pin_mem)
			
		v_ls = M_deepvo.get_loss(v_x, v_y).data.cpu().numpy()
		v_loss_list.append(float(v_ls))
		loss_mean_valid += float(v_ls)
	print('Testing time: {:.1f} [s]'.format(time.time()-st_t))
	loss_mean_valid /= len(valid_dl)

	f = open(par.record_path, 'a')
	f.write('Epoch {}\ntrain loss mean: {}, std: {:.2f}\nvalid loss mean: {}, std: {:.2f}\n'.format(ep+1, loss_mean, np.std(t_loss_list), loss_mean_valid, np.std(v_loss_list)))
	
	print('Epoch {}\ntrain loss mean: {}, std: {:.2f}\nvalid loss mean: {}, std: {:.2f}\n'.format(ep+1, loss_mean, np.std(t_loss_list), loss_mean_valid, np.std(v_loss_list)))

	# Save model
	# save if the valid loss decrease
	check_interval = 1
	if loss_mean_valid < min_loss_v and ep % check_interval == 0:
		min_loss_v = loss_mean_valid
		print('Save model at ep {}, mean of valid loss: {}'.format(ep+1, loss_mean_valid))
		torch.save(M_deepvo.state_dict(),	par.save_model_path		+ '.valid')
		torch.save(optimizer.state_dict(),	par.save_optimzer_path	+ '.valid')
	
	# save if the training loss decrease
	check_interval = 1
	if loss_mean < min_loss_t and ep % check_interval == 0:
		min_loss_t = loss_mean
		print('Save model at ep {}, mean of train loss: {}'.format(ep+1, loss_mean))
		torch.save(M_deepvo.state_dict(), 	par.save_model_path		+ '.train')
		torch.save(optimizer.state_dict(), 	par.save_optimzer_path	+ '.train')
	
	f.close()
