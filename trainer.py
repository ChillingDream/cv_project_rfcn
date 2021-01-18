import torch
from torch import nn
from torch.nn import functional as F
from models.rpn import RPNTargetGenerator, RoITargetGenerator
from torchnet.meter import ConfusionMeter, AverageValueMeter
from visualizer import Visualizer


class Trainer(nn.Module):
	def __init__(self, rfcn, config):
		super().__init__()
		self.rfcn = rfcn
		self.rpn_target_generator = RPNTargetGenerator()
		self.roi_target_generator = RoITargetGenerator()
		self.rpn_sigma = config.rpn_sigma
		self.roi_sigma = config.roi_sigma
		self.loc_normalize_mean = nn.Parameter(torch.tensor((0., 0., 0., 0.)), False)
		self.loc_normalize_std = nn.Parameter(torch.tensor((.1, .1, .2, .2)), False)
		self.rpn_cm = ConfusionMeter(2)
		self.roi_cm = ConfusionMeter(config.num_classes)
		self.loss_avgmeter = {k: AverageValueMeter() for k in
							  ['rpn_loc_loss', 'rpn_fg_loss', 'roi_loc_loss', 'roi_cls_loss', 'total_loss']}
		self.optimizer = self._get_optimizer(config)
		self.vis = Visualizer(env=config.exp_name)
		self.train()

	def forward(self, imgs, bboxes, labels, scale):
		"""
		Args:
			imgs: (N, C, H, W)
			bboxes: (N, R, 4)
			labels: (N, R)
			scale: scale factor of preprocessing
		"""
		if imgs.size(0) != 1:
			raise ValueError("Only batch_size 1 is supported.")
		img_size = imgs.size()[2:]

		features = self.rfcn.extractor(imgs)
		rpn_scores, rpn_locs, rois, roi_indices, anchors = self.rfcn.rpn(features, img_size)

		bbox = bboxes[0]
		label = labels[0]
		rpn_locs = rpn_locs[0]
		rpn_scores = rpn_scores[0]

		sample_roi, gt_roi_loc, gt_roi_label = self.roi_target_generator(
			rois, bbox, label, self.loc_normalize_mean, self.loc_normalize_std
		)
		roi_score, roi_loc = self.rfcn.roi_head(features, sample_roi, torch.zeros(len(sample_roi), device="cuda:0"))

		# RPN losses
		gt_rpn_locs, gt_rpn_labels = self.rpn_target_generator(anchors, bboxes[0], img_size, self.vis)
		rpn_loc_loss = _loc_loss(rpn_locs, gt_rpn_locs, gt_rpn_labels, self.rpn_sigma)
		rpn_fg_loss = F.cross_entropy(rpn_scores, gt_rpn_labels, ignore_index=-1)
		self.rpn_cm.add(rpn_scores[gt_rpn_labels > -1].detach(), gt_rpn_labels[gt_rpn_labels > -1].detach())

		# RoI losses
		n_smaple = roi_loc.size(0)
		roi_loc = roi_loc.view(n_smaple, -1, 4)
		roi_loc = roi_loc[torch.arange(n_smaple), gt_roi_label].contiguous()
		roi_loc_loss = _loc_loss(roi_loc, gt_roi_loc, gt_roi_label, self.roi_sigma)
		roi_cls_loss = F.cross_entropy(roi_score, gt_roi_label)
		self.roi_cm.add(roi_score.cpu().detach().clone(), gt_roi_label.cpu())

		total_loss = rpn_loc_loss + rpn_fg_loss + roi_loc_loss + roi_cls_loss
		#total_loss = rpn_loc_loss + rpn_fg_loss
		return {'rpn_loc_loss': rpn_loc_loss,
				'rpn_fg_loss': rpn_fg_loss,
				'roi_loc_loss': roi_loc_loss,
				'roi_cls_loss': roi_cls_loss,
				'total_loss': total_loss}

	def train_step(self, imgs, bboxes, labels, scale):
		self.optimizer.zero_grad()
		losses = self.forward(imgs, bboxes, labels, scale)
		for k, v in losses.items():
			self.loss_avgmeter[k].add(v.cpu().detach())
		losses['total_loss'].backward()
		self.optimizer.step()
		return losses

	def save(self, save_path):
		torch.save({'model': self.rfcn.state_dict(),
					'optimizer': self.optimizer.state_dict()}, save_path)

	def load(self, save_path):
		checkpoint = torch.load(save_path)
		self.rfcn.load_state_dict(checkpoint['model'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])

	def reset_meters(self):
		for meter in self.loss_avgmeter.values():
			meter.reset()
		self.rpn_cm.reset()
		self.roi_cm.reset()

	def get_meter(self):
		return {k: v.value()[0] for k, v in self.loss_avgmeter.items()}

	def _get_optimizer(self, config):
		lr = config.lr
		params = []
		for key, value in dict(self.rfcn.named_parameters()).items():
			if value.requires_grad:
				if 'bias' in key:
					params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
				else:
					params += [{'params': [value], 'lr': lr, 'weight_decay': config.weight_decay}]
		if config.optimizer == 'adam':
			opt = torch.optim.Adam(params)
		elif config.optimizer == 'sgd':
			opt = torch.optim.SGD(params, momentum=0.9)
		else:
			raise ValueError("optimizer should either be adam or sgd")
		return opt


def _smooth_l1_loss(x, t, weight, sigma):
	sigma2 = sigma ** 2
	diff = ((x - t) * weight).abs()
	smooth = (diff < 1 / sigma2).float()
	loss = smooth * sigma2 * diff ** 2 / 2 + (1 - smooth) * (diff - 0.5 / sigma2)
	return loss.sum()


def _loc_loss(locs, gt_locs, gt_labels, sigma):
	return _smooth_l1_loss(locs, gt_locs, (gt_labels > 0).float().unsqueeze(1), sigma) / (gt_labels >= 0).sum().float()
