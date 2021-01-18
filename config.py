import torch

class Config:
	exp_name = 'default'
	device = torch.device('cuda:0')
	load_path = None
	save_path = exp_name + '.pt'

	num_classes = 21

	rpn_sigma = 3.
	roi_sigma = 1.

	optimizer = 'sgd'
	batch_size = 1
	lr = 1e-3
	lr_decay = 0.1
	weight_decay = 5e-4
	epoch = 20

	eval_iters = 100


config = Config()
