from models.rfcn import RFCN
from trainer import Trainer
from config import config
from tqdm import tqdm
from eval_voc_utils import eval_detection_voc


def evaluate(rfcn, dataloader, test_num=10000):
	pred_bboxes, pred_labels, pred_scores = [], [], []
	gt_bboxes, gt_labels, gt_difficulties = [], [], []
	for i, (imgs, bboxes, labels, difficulties) in tqdm(enumerate(dataloader)):
		gt_bboxes += list(bboxes.numpy())
		gt_labels += list(labels.numpy())
		gt_difficulties += list(difficulties.numpy())
		bboxes, labels, scores = rfcn.predict(imgs)
		pred_bboxes += list(bboxes.numpy())
		pred_labels += list(labels.numpy())
		pred_scores += list(scores.numpy())
		if i == test_num - 1:
			break

	return eval_detection_voc(
		pred_bboxes, pred_labels, pred_scores,
		gt_bboxes, gt_labels, gt_difficulties,
		use_07_metric=False
	)


def train():
	print('loading data.')
	train_data, val_data, test_data = get_dataloader('VOC')
	print('building model.')
	rcfn = RFCN(config)
	trainer = Trainer(rcfn, config)
	trainer.to(config.device)
	best_map = 0

	print('training start')
	for epoch in range(config.epoch):
		trainer.reset_meters()
		for iters, batch in tqdm(enumerate(train_data)):
			imgs, bboxes, labels, scale = map(lambda x: x.to(config.device), batch)
			trainer.train_step(imgs, bboxes, labels, scale)
			if (iters + 1) % config.eval_iters == 0:
				trainer.vis.plot(trainer.get_meter())
				trainer.vis.show_image('gt_img', imgs)

			bboxes, labels, scores = trainer.rfcn.predict(imgs)
			trainer.vis.show_image_bbox('pred_img', bboxes[0], labels[0], scores[0])
			trainer.vis.text(str(trainer.rpn_cm.value().tolist()), 'rpn_cm')
			trainer.vis.show_image('roi_cm', trainer.roi_cm.conf)

		eval_result = evaluate(trainer.rfcn, val_data)
		trainer.vis.plot('val_map', eval_result['map'])

		if eval_result['map'] > best_map:
			best_map = eval_result['map']
			trainer.save(config.save_path)


def get_dataloader(data_name='VOC'):
	raise NotImplementedError


if __name__ == '__main__':
	train()
