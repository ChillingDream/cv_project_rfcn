import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from tqdm import tqdm

from models.rfcn import RFCN
from trainer import Trainer
from config import config
from Pascal_VOC_dataset import Pascal_VOC_dataset
from BDD10K_dataset import BDD10K_dataset
from torch.utils.data import DataLoader
from test import evaluate


def train(dataset='BDD'):
	print('loading data.')
	train_data, val_data = get_dataloader(dataset)
	print('building model.')
	rfcn = RFCN(config, backbone='vgg16', head='rcnn', class_specific=False)
	trainer = Trainer(rfcn, config).cuda()
	best_map = 0

	print('training start')
	for epoch in range(config.epoch):
		trainer.reset_meters()
		for iters, batch in enumerate(tqdm(train_data)):
			if dataset == 'BDD':
				imgs, bboxes, labels, scale = map(lambda x: x.to(config.device), batch)
			else:
				imgs, bboxes, labels, scale, difficults = map(lambda x: x.to(config.device), batch)
			trainer.train_step(imgs, bboxes, labels, scale)
			# if iters == 20:
			# 	break
			if (iters + 1) % config.eval_iters == 0:
				trainer.vis.multi_plot(trainer.get_meter())

				img = Pascal_VOC_dataset.inverse_normalize(imgs[0].cpu().numpy())
				trainer.vis.show_image_bbox('gt_img', img, *map(lambda x: x.cpu().numpy(), [bboxes[0], labels[0]]))

				bboxes, labels, scores = trainer.rfcn.predict(imgs, vis=True)
				trainer.vis.show_image_bbox('pred_img', img, *map(lambda x: x.cpu().numpy(), [bboxes, labels, scores]))
				trainer.vis.text(str(trainer.rpn_cm.value().tolist()), 'rpn_cm')
				trainer.vis.show_image('roi_cm', np.array(trainer.roi_cm.conf, dtype=np.float))
		eval_result = evaluate(trainer.rfcn, val_data, dataset)
		trainer.vis.plot('val_map', eval_result['map'])

		if eval_result['map'] > best_map:
			best_map = eval_result['map']
			print('new best map:', best_map)
			trainer.save(config.save_path)

		if epoch + 1 == config.pre_epoch:
			trainer.load(config.save_path)
			trainer.scale_lr(config.lr_decay)

	print('training done')


def get_dataloader(data_name):
	# raise NotImplementedError
	if data_name == 'VOC':
		train_dataset = Pascal_VOC_dataset(devkit_path = 'VOCdevkit', dataset_list = ['2007_trainval']) # Remember to change the path!
		val_dataset = Pascal_VOC_dataset(devkit_path = 'VOCdevkit', dataset_list = ['2007_test'])
	elif data_name == 'BDD':
		train_dataset = BDD10K_dataset(load_from='/home/zkj/codes/cv_project_rfcn/bdd100k_small_train.pkl') # Remember to change the path!
		val_dataset = BDD10K_dataset(load_from='/home/zkj/codes/cv_project_rfcn/bdd100k_small_val.pkl') # Remember to change the path!
	train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
	val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
	return train_loader, val_loader


if __name__ == '__main__':
	train(config.dataset)
