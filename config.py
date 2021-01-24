from argparse import ArgumentParser
import torch

class Config:
	exp_name = 'BDD'
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

	train_dump_path = ''
	val_dump_path = ''
	bdd100k_path = ''

	def _parse(self):
		parser = ArgumentParser()
		parser.add_argument('dataset', choices=['VOC', 'BDD'])
		parser.add_argument("-b", "--bdd100k_path", type=str, default='')
		parser.add_argument("-t", "--train_dump_path", type=str, default='')
		parser.add_argument("-b", "--val_dump_path", type=str, default='')
		args = parser.parse_args()
		self.dataset = args.dataset
		self.train_dump_path = args.train_dump_path
		self.val_dump_path = args.val_dump_path
		self.bdd100k_path = args.bdd100k_path


config = Config()
config._parse()
