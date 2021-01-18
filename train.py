import os
import numpy as np
from models.rfcn import RFCN
from trainer import Trainer
from config import config
from tqdm import tqdm
from eval_voc_utils import eval_detection_voc
from Pascal_VOC_dataset import Pascal_VOC_dataset
from torch.utils.data import Dataset, DataLoader, TensorDataset

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def evaluate(rfcn, dataloader, test_num=10000):
	pred_bboxes, pred_labels, pred_scores = [], [], []
	gt_bboxes, gt_labels, gt_difficults = [], [], []
	for i, (imgs, bboxes, labels, scale, difficults) in enumerate(tqdm(dataloader)):
		gt_bboxes += list(bboxes.numpy())
		gt_labels += list(labels.numpy())
		gt_difficults += list(difficults.numpy())
		bboxes, labels, scores = rfcn.predict(imgs.to(config.device))
		pred_bboxes += [bboxes.cpu().numpy()]
		pred_labels += [labels.cpu().numpy()]
		pred_scores += [scores.cpu().numpy()]
		if i == test_num - 1 or i == 1:
			break

	return eval_detection_voc(
		pred_bboxes, pred_labels, pred_scores,
		gt_bboxes, gt_labels, gt_difficults,
		use_07_metric=False
	)


def train():
	print('loading data.')
	train_data, val_data = get_dataloader('VOC')
	print('building model.')
	rcfn = RFCN(config, backbone='vgg16', head='rcnn').to(config.device)
	trainer = Trainer(rcfn, config)
	trainer.to(config.device)
	best_map = 0

	print('training start')
	for epoch in range(config.epoch):
		trainer.reset_meters()
		for iters, batch in enumerate(tqdm(train_data)):
			break
			imgs, bboxes, labels, scale, difficults = map(lambda x: x.to(config.device), batch)
			trainer.train_step(imgs, bboxes, labels, scale)
			if (iters + 1) % config.eval_iters == 0:
				trainer.vis.multi_plot(trainer.get_meter())

				img = Pascal_VOC_dataset.inverse_normalize(imgs[0].cpu().numpy())
				trainer.vis.show_image_bbox('gt_img', img, *map(lambda x: x.cpu().numpy(), [bboxes[0], labels[0]]))

				bboxes, labels, scores = trainer.rfcn.predict(imgs, vis=True)
				#index = np.argsort(rpn_fg_scores[0].cpu().numpy())[-10:]
				#trainer.vis.show_image_bbox('pred_rpn', img, rpn_boxes[0][index].cpu().numpy(), score=rpn_fg_scores[0][index].cpu().numpy())
				trainer.vis.show_image_bbox('pred_img', img, *map(lambda x: x.cpu().numpy(), [bboxes, labels, scores]))
				trainer.vis.text(str(trainer.rpn_cm.value().tolist()), 'rpn_cm')
				trainer.vis.show_image('roi_cm', np.array(trainer.roi_cm.conf, dtype=np.float))

		eval_result = evaluate(trainer.rfcn, val_data)
		trainer.vis.plot('val_map', eval_result['map'])

		if eval_result['map'] > best_map:
			best_map = eval_result['map']
			print('new best map:', best_map)
			trainer.save(config.save_path)

		if epoch == 9:
			trainer.load(config.save_path)

	print('training done')


def get_dataloader(data_name='VOC'):
	# raise NotImplementedError
	voc_train_dataset = Pascal_VOC_dataset(devkit_path = 'VOCdevkit', dataset_list = ['2007_trainval'], load_all=False) # Remember to change the path!
	voc_val_dataset = Pascal_VOC_dataset(devkit_path = 'VOCdevkit', dataset_list = ['2007_test'], load_all=False)
	#voc_test_dataset = Pascal_VOC_dataset(devkit_path = 'VOCdevkit', dataset_list = ['2007_test'], load_all=True)
	train_loader = DataLoader(dataset=voc_train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
	val_loader = DataLoader(dataset=voc_val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)
	#test_loader = DataLoader(dataset=voc_test_dataset, batch_size=config.batch_size, shuffle=False)
	return train_loader, val_loader


if __name__ == '__main__':
	train()
