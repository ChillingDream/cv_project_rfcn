import torch


class Config:
	device = torch.device('cuda:0')
	load_path = None
	save_path = './'

	num_classes = 2

	rpn_sigma = 3.
	roi_sigma = 1.

	lr = 0.001
	weight_decay = 0.1
	epoch = 10

	eval_iters = 100


config = Config()
