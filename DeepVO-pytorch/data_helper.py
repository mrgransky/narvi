import os, glob, torch, time, sys
from tqdm import tqdm 
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from params import par
from helper import normalize_angle_delta

def generate_data(sequences, seq_len_range, overlap, ts=1, 
							pad_y=False, shuffle=False, sort=True):
	X_path, Y, X_len = [], [], []
	for seq in sequences:
		start_t = time.time()
		try:
			poses 	= np.load('{}{}.npy'.format(par.pose_dir, seq))
			fpaths 	= glob.glob('{}{}/*.png'.format(par.image_dir, seq))
			print "seq[{}] has {} GT_frames & {} img_frames".format(seq, len(poses), len(fpaths))
			fpaths.sort()
		except Exception as e:
			print str(e)
			sys.exit()
			
		# Fixed seq_len
		if seq_len_range[0] == seq_len_range[1]:
			if ts > 1:
				sample_interval = int(np.ceil(seq_len_range[0] / ts))
				start_frames = list(range(0, seq_len_range[0], sample_interval))
				print('Sample start from frame {}'.format(start_frames))
			else:
				start_frames = [0]
			
			for st in start_frames:
				seq_len = seq_len_range[0]
				n_frames = len(fpaths) - st
				jump = seq_len - overlap
				res = n_frames % seq_len
				
				if res != 0:
					n_frames = n_frames - res
				
				x_segs = [fpaths[i:i+seq_len] for i in range(st, n_frames, jump)]
				y_segs = [poses[i:i+seq_len] for i in range(st, n_frames, jump)]
				Y += y_segs
				X_path += x_segs
				X_len += [len(xs) for xs in x_segs]
		# Random segment to sequences with diff lengths
		else:
			assert(overlap < min(seq_len_range))
			n_frames = len(fpaths)
			min_len, max_len = seq_len_range[0], seq_len_range[1]
			for i in range(ts):
				start = 0
				while True:
					n = np.random.random_integers(min_len, max_len)
					if start + n < n_frames:
						x_seg = fpaths[start:start+n]
						X_path.append(x_seg)
						if not pad_y:
							Y.append(poses[start:start+n])
						else:
							pad_zero = np.zeros((max_len-n, 15))
							padded = np.concatenate((poses[start:start+n], pad_zero))
							Y.append(padded.tolist())
					else:
						print('Last %d frames is not used' %(start+n-n_frames))
						break
					start += n - overlap
					X_len.append(len(x_seg))
		print('seq {} finish in {} sec'.format(seq, time.time()-start_t))
	
	# Convert to pandas dataframes
	data = {'seq_len': X_len, 'image_path': X_path, 'pose': Y}
	df = pd.DataFrame(data, columns = ['seq_len', 'image_path', 'pose'])
	# Shuffle through all videos
	if shuffle:
		df = df.sample(frac=1)
	# Sort dataframe by seq_len
	if sort:
		df = df.sort_values(by=['seq_len'], ascending=False)
	return df

def get_partition_data_info(partition, sequences, seq_len_range, overlap, ts=1, pad_y=False, shuffle=False, sort=True):
    X_path = [[], []]
    Y = [[], []]
    X_len = [[], []]
    df_list = []

    for part in range(2):
        for seq in sequences:
            start_t = time.time()
            poses = np.load('{}{}.npy'.format(par.pose_dir, seq))  # (n_images, 6)
            fpaths = glob.glob('{}{}/*.png'.format(par.image_dir, seq))
            fpaths.sort()


            # Get the middle section as validation set
            n_val = int((1-partition)*len(fpaths))
            st_val = int((len(fpaths)-n_val)/2)
            ed_val = st_val + n_val
            print('st_val: {}, ed_val:{}'.format(st_val, ed_val))
            if part == 1:
                fpaths = fpaths[st_val:ed_val]
                poses = poses[st_val:ed_val]
            else:
                fpaths = fpaths[:st_val] + fpaths[ed_val:]
                poses = np.concatenate((poses[:st_val], poses[ed_val:]), axis=0)

            # Random Segment
            assert(overlap < min(seq_len_range))
            n_frames = len(fpaths)
            min_len, max_len = seq_len_range[0], seq_len_range[1]
            for i in range(ts):
                start = 0
                while True:
                    n = np.random.random_integers(min_len, max_len)
                    if start + n < n_frames:
                        x_seg = fpaths[start:start+n] 
                        X_path[part].append(x_seg)
                        if not pad_y:
                            Y[part].append(poses[start:start+n])
                        else:
                            pad_zero = np.zeros((max_len-n, 6))
                            padded = np.concatenate((poses[start:start+n], pad_zero))
                            Y[part].append(padded.tolist())
                    else:
                        print('Last %d frames is not used' %(start+n-n_frames))
                        break
                    start += n - overlap
                    X_len[part].append(len(x_seg))
            print('seq {} finish in {} sec'.format(seq, time.time()-start_t))
        
        # Convert to pandas dataframes
        data = {'seq_len': X_len[part], 'image_path': X_path[part], 'pose': Y[part]}
        df = pd.DataFrame(data, columns = ['seq_len', 'image_path', 'pose'])
        # Shuffle through all videos
        if shuffle:
            df = df.sample(frac=1)
        # Sort dataframe by seq_len
        if sort:
            df = df.sort_values(by=['seq_len'], ascending=False)
        df_list.append(df)
    return df_list

class SortedRandomBatchSampler(Sampler):
    def __init__(self, info_dataframe, batch_size, drop_last=False):
        self.df = info_dataframe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.unique_seq_lens = sorted(self.df.iloc[:].seq_len.unique(), reverse=True)
        # Calculate len (num of batches, not num of samples)
        self.len = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            self.len += n_batch

    def __iter__(self):
        # Calculate number of sameples in each group (grouped by seq_len)
        list_batch_indexes = []
        start_idx = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            rand_idxs = (start_idx + torch.randperm(n_sample)).tolist()
            tmp = [rand_idxs[s*self.batch_size: s*self.batch_size+self.batch_size] for s in range(0, n_batch)]
            list_batch_indexes += tmp
            start_idx += n_sample
        return iter(list_batch_indexes)

    def __len__(self):
        return self.len


class LoadMyDataset(Dataset):
	def __init__(self, info_dataframe, resize_mode='crop', new_sizeize=None, 
							img_mean=None, img_std=(1,1,1), minus_point_5=False):
		# Transforms
		tf = []
		if resize_mode == 'crop':
			tf.append(transforms.CenterCrop((new_sizeize[0], new_sizeize[1])))
		elif resize_mode == 'rescale':
			tf.append(transforms.Resize((new_sizeize[0], new_sizeize[1])))
		
		tf.append(transforms.ToTensor())
		#tf.append(transforms.Normalize(mean=img_mean, std=img_std))
		
		self.transformer 		= transforms.Compose(tf)
		self.minus_point_5 	= minus_point_5
		self.normalizer 		= transforms.Normalize(mean=img_mean, std=img_std)
		
		self.data_info 	= info_dataframe
		
		self.seq_len_list = list(self.data_info.seq_len)
		self.IMG_arr		= np.asarray(self.data_info.image_path)  # image paths
		self.GT_arr 		= np.asarray(self.data_info.pose)
		
	def __getitem__(self, index):
		raw_GT = np.hsplit(self.GT_arr[index], np.array([6]))
		GT_seq = raw_GT[0]
		GT_rot = raw_GT[1][0].reshape((3, 3)).T # opposite rot of 1st frame
		
		GT_seq = torch.FloatTensor(GT_seq)
		
		# get relative pose w.r.t. previois frame 
		# GT_seq[1:] = GT_seq[1:] - GT_seq[0:-1]
		
		# get relative pose w.r.t. 1st frame in the sequence
		GT_seq[1:] = GT_seq[1:] - GT_seq[0]
		
		#print('Item before transform: ' + str(index) + '   ' + str(GT_seq))
		
		# here we rotate the sequence relative to the first frame
		for sq in GT_seq[1:]:
			location = torch.FloatTensor(GT_rot.dot(sq[3:].numpy()))
			sq[3:] = location[:]
			#print "location = {}".format(location)
			
		# get relative pose w.r.t. previous frame
		GT_seq[2:] = GT_seq[2:] - GT_seq[1:-1]
		
		# rotation angles over Y axis go through PI -PI discontinuity
		for sq in GT_seq[1:]:
			sq[0] = normalize_angle_delta(sq[0])
		
		#print('Item after transform: ' + str(index) + '   ' + str(GT_seq))
		image_path_sequence 	= self.IMG_arr[index]
		seq_len 					= torch.tensor(self.seq_len_list[index])
		#seq_len 				= torch.tensor(len(image_path_sequence))
		#print "len(seq): {}".format(seq_len)
		#print "img_path:\t{}".format(image_path_sequence)
			
		IMG_seq = []
		for img_path in image_path_sequence:
		#for img_path in tqdm(image_path_sequence):
			#print "img_path:\t{}".format(img_path)
			try:
				img_as_img = Image.open(img_path)
			except Exception as e:
				print str(e)
				pass
			img_as_tensor = self.transformer(img_as_img)
			if self.minus_point_5:
				img_as_tensor = img_as_tensor - 0.5  # from [0, 1] -> [-0.5, 0.5]
			img_as_tensor = self.normalizer(img_as_tensor)
			img_as_tensor = img_as_tensor.unsqueeze(0)
			IMG_seq.append(img_as_tensor)
		IMG_seq = torch.cat(IMG_seq, 0)
		return (seq_len, IMG_seq, GT_seq)
		
	def __len__(self):
		return len(self.data_info.index)

# Example of usage
if __name__ == '__main__':
    start_t = time.time()
    # Gernerate info dataframe
    overlap = 1
    ts = 1
    sequences = ['00']
    #sequences = ['04']
    seq_len_range = [5, 7]
    df = generate_data(sequences, seq_len_range, overlap, ts)
    print('Elapsed Time (generate_data): {} sec'.format(time.time()-start_t))
    
    # Customized Dataset, Sampler
    n_workers = 4
    resize_mode = 'crop'
    new_size = (150, 600)
    img_mean = (-0.14968217427134656, -0.12941663107068363, -0.1320610301921484)
    
    dataset = ImageSequenceDataset(df, resize_mode, new_size, img_mean)
    
    sorted_sampler = SortedRandomBatchSampler(df, batch_size=4, drop_last=True)
    
    dataloader = DataLoader(dataset, batch_sampler=sorted_sampler, n_workers=n_workers)
    
    print('Elapsed Time (dataloader): {} sec'.format(time.time()-start_t))

    for batch in dataloader:
        s, x, y = batch
        print('='*50)
        print('len:{}\nx:{}\ny:{}'.format(s, x.shape, y.shape))
    
    print('Elapsed Time: {} sec'.format(time.time()-start_t))
    print('Number of workers = ', n_workers)
