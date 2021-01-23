from argparse import ArgumentParser
import torch

class Config:
	exp_name = 'BDD_100K_2'
	device = torch.device('cuda:0')
	load_path = None
	save_path = exp_name + '.pt'

	dataset = 'BDD'
	num_classes = 21

	rpn_sigma = 3.
	roi_sigma = 1.
	loc_loss_lambda = 1

	optimizer = 'sgd'
	batch_size = 1
	lr = 1e-3
	lr_decay = 0.1
	weight_decay = 5e-4
	pre_epoch = 10
	epoch = 15

	eval_iters = 100

	def _parse(self):
		parser = ArgumentParser()
		parser.add_argument('dataset', choices=['VOC', 'BDD'])
		args = parser.parse_args()
		config.dataset = args.dataset


config = Config()
config._parse()
