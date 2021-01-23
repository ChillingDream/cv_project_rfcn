import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from BDD100K_dataset import BDD100K_dataset
from Pascal_VOC_dataset import Pascal_VOC_dataset
from config import config
from eval_voc_utils import eval_detection_voc
from models.rfcn import RFCN


def evaluate(rfcn, dataloader, dataset, test_num=10000):
	pred_bboxes, pred_labels, pred_scores = [], [], []
	gt_bboxes, gt_labels, gt_difficults = [], [], []
	if dataset == 'BDD':
		gt_difficults = None
	for i, pack in enumerate(tqdm(dataloader)):
		if len(pack) == 5:
			imgs, bboxes, labels, scale, difficults = pack
		else:
			imgs, bboxes, labels, scale = pack
		gt_bboxes += list(bboxes.numpy())
		gt_labels += list(labels.numpy())
		if dataset == 'VOC':
			gt_difficults += list(difficults.numpy())
		bboxes, labels, scores = rfcn.predict(imgs.to(config.device))
		pred_bboxes += [bboxes.cpu().numpy()]
		pred_labels += [labels.cpu().numpy()]
		pred_scores += [scores.cpu().numpy()]
		if i == test_num - 1:
			break

	return eval_detection_voc(
		pred_bboxes, pred_labels, pred_scores,
		gt_bboxes, gt_labels, gt_difficults,
		use_07_metric=False
	)


def test(dataset='BDD'):
	test_data = get_dataloader(dataset)
	rfcn = RFCN(config, backbone='vgg16', head='rcnn', class_specific=False)
	rfcn.load_state_dict(torch.load(config.exp_name + '.pt')['model'])
	eval_result = evaluate(rfcn, test_data, dataset)
	print(eval_result)


def get_dataloader(data_name='BDD'):
	# raise NotImplementedError
	if data_name == 'VOC':
		test_dataset = Pascal_VOC_dataset(devkit_path = 'VOCdevkit', dataset_list = ['2007_test'])
	elif data_name == 'BDD':
		test_dataset = BDD100K_dataset(load_from='/home/zkj/codes/cv_project_rfcn/bdd100k_small_test.pkl') # Remember to change the path!
	test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
	return test_loader


if __name__ == '__main__':
	test(config.dataset)
