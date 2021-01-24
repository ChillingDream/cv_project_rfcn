import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from BDD100K_dataset import BDD100K_dataset
from Pascal_VOC_dataset import Pascal_VOC_dataset
from config import config
from eval_voc_utils import eval_detection_voc
from models.rfcn import RFCN
from visualizer import Visualizer


def evaluate(rfcn, dataloader, dataset, test_num=10000, vis=None):
	pred_bboxes, pred_labels, pred_scores = [], [], []
	gt_bboxes, gt_labels, gt_difficults = [], [], []
	if dataset == 'BDD':
		gt_difficults = None
	class_zero_cnt = 0
	all_cnt = 0
	for i, pack in enumerate(tqdm(dataloader)):
		if len(pack) == 5:
			imgs, bboxes, labels, scale, difficults = pack
		else:
			imgs, bboxes, labels, scale = pack
		# print(bboxes[0].size())
		# print(labels[0].size())
		# bboxes[0] = bboxes[0][labels[0] != 0]
		# print(((~labels[0].eq(0)).unsqueeze(1)).size())
		# new_bboxes = torch.masked_select(bboxes[0], (~labels[0].eq(0)).unsqueeze(1))
		new_bboxes = bboxes[0][labels[0] != 0]
		# imgs = imgs[labels[0] != 0]
		# labels[0] = labels[0][labels[0] != 0]
		new_labels = torch.masked_select(labels[0], ~labels[0].eq(0))
		# print(bboxes[0].size())
		# print(labels[0].size())
		# print('done')
		# gt_bboxes += list(bboxes.numpy())
		# gt_labels += list(labels.numpy())
		gt_bboxes += list(new_bboxes.unsqueeze(0).numpy())
		gt_labels += list(new_labels.unsqueeze(0).numpy())
		class_zero_cnt += len(labels[0].cpu().numpy()[labels[0].cpu().numpy() == 0])
		all_cnt += len(labels[0].cpu().numpy())
		if i%30 ==0 and vis is not None:
			# print(labels[0].cpu().numpy())
			img = BDD100K_dataset.inverse_normalize(imgs[0].cpu().numpy())
			# print(new_bboxes.size())
			vis.show_image_bbox('gt_img', img, *map(lambda x: x.cpu().numpy(), [new_bboxes, new_labels]))
			# vis.show_image_bbox('pred_img', img, *map(lambda x: x.cpu().numpy(), [bboxes, labels, scores]))
		if dataset == 'VOC':
			gt_difficults += list(difficults.numpy())
		bboxes, labels, scores = rfcn.predict(imgs.to(config.device), vis=True)
		# bboxes = bboxes[labels[0] != 0]
		# labels = labels[labels[0] != 0]
		pred_bboxes += [bboxes.cpu().numpy()]
		pred_labels += [labels.cpu().numpy()]
		pred_scores += [scores.cpu().numpy()]
		if i%30 ==0 and vis is not None:
			# img = BDD100K_dataset.inverse_normalize(imgs[0].cpu().numpy())
			# vis.show_image_bbox('gt_img', img, *map(lambda x: x.cpu().numpy(), [bboxes[0], labels[0]]))
			vis.show_image_bbox('pred_img', img, *map(lambda x: x.cpu().numpy(), [bboxes, labels, scores]))
		if i == test_num - 1:
			break
	print('class_zero_cnt:', class_zero_cnt)
	print('all_cnt:', all_cnt)
	return eval_detection_voc(
		pred_bboxes, pred_labels, pred_scores,
		gt_bboxes, gt_labels, gt_difficults,
		use_07_metric=False
	)


def test(dataset='BDD'):
	test_data = get_dataloader(dataset)
	rfcn = RFCN(config, backbone='vgg16', head='rcnn', class_specific=False).cuda()
	rfcn.load_state_dict(torch.load(config.save_path)['model'])
	vis = Visualizer(env=config.exp_name)
	eval_result = evaluate(rfcn, test_data, dataset, vis=vis)
	print(eval_result)


def get_dataloader(data_name='BDD'):
	# raise NotImplementedError
	if data_name == 'VOC':
		test_dataset = Pascal_VOC_dataset(devkit_path = 'VOCdevkit', dataset_list = ['2007_test'], just_car=config.just_car)
	elif data_name == 'BDD':
		# test_dataset = BDD100K_dataset(load_from='/home/zkj/codes/cv_project_rfcn/bdd100k_small_val2.pkl') # Remember to change the path!
		if config.val_dump_path:
			test_dataset = BDD100K_dataset(load_from=config.val_dump_path) # Remember to change the path!
		else:
			test_dataset = BDD100K_dataset(bdd100k_path=config.bdd100k_path, dataset_list=['val'], just_car=config.just_car)
	test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
	return test_loader


if __name__ == '__main__':
	test(config.dataset)
