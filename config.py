import torch


class Config:
	device = torch.device('cuda:0')
	load_path = None
	save_path = './'

	num_classes = 21

	rpn_sigma = 3.
	roi_sigma = 1.

	batch_size = 1
	lr = 0.001
	weight_decay = 0.1
	epoch = 10

	eval_iters = 1


config = Config()
