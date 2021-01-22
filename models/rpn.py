import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms, box_iou

from utils import generator_anchor, enumerate_shifted_anchors, loc2bbox, bbox2loc


class RPN(nn.Module):
	"""Region Proposal Network

	Args:
		in_channels: (int).
		mid_channels: (int).
		ratios: tuple. ratios of width to height of the anchors.
		anchor_scales: tuple. the area of anchors.
		feat_stride: (int). stride size after extractor.
	"""

	def __init__(self, in_channels, mid_channels, ratios=(0.5, 1, 2), anchor_scales=(8, 16, 32), feat_stride=32):
		super().__init__()
		self.anchors = generator_anchor(16, ratios, anchor_scales)
		self.feat_stride = feat_stride

		n_anchors = self.anchors.size(0)
		self.conv_layer = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
		self.score = nn.Conv2d(mid_channels, n_anchors * 2, 1, 1, 0)
		self.loc = nn.Conv2d(mid_channels, n_anchors * 4, 1, 1, 0)

		nn.init.normal_(self.conv_layer.weight, 0, 0.01)
		nn.init.normal_(self.score.weight, 0, 0.01)
		nn.init.normal_(self.loc.weight, 0, 0.01)

		self.proposal_layer = ProposalLayer()

	def forward(self, x, img_size):
		"""
		Args:
			x: (N, C, H, W)
			img_size: tuple (H, W). the size of original image.
		"""
		batch_size = x.size(0)
		shifted_anchors = enumerate_shifted_anchors(self.anchors, self.feat_stride,	x.size(2), x.size(3)).cuda()

		h = F.relu(self.conv_layer(x))

		rpn_locs = self.loc(h)
		rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

		rpn_scores = self.score(h)
		rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
		rpn_fg_scores = F.softmax(rpn_scores, 2)[:, :, 1]

		rois = []
		roi_indices = []
		for i in range(batch_size):
			roi = self.proposal_layer(
				rpn_locs[i].detach(),
				rpn_fg_scores[i].detach(),
				shifted_anchors,
				img_size,
				self.training,
			)
			rois.append(roi)
			roi_indices.append(i * torch.ones(len(roi), dtype=torch.long).cuda())
		rois = torch.cat(rois, 0)
		roi_indices = torch.cat(roi_indices, 0)

		return rpn_scores, rpn_locs, rois, roi_indices, shifted_anchors


class ProposalLayer:
	def __init__(self,
				 nms_threshold=0.7,
				 n_in_train=12000,
				 n_out_train=2000,
				 n_in_test=6000,
				 n_out_test=300,
				 min_size=16):
		self.nms_threshold = nms_threshold
		self.n_in_train = n_in_train
		self.n_out_train = n_out_train
		self.n_in_test = n_in_test
		self.n_out_test = n_out_test
		self.min_size = min_size

	def __call__(self, locs, scores, anchors, img_size, training):
		"""
		Args:
			locs: (R, 4)
			scores: (R, 2)
			anchors: (R, 4)
			img_size: tuple (H, W).
		"""
		n_in, n_out = (self.n_in_train, self.n_out_train) if training else (self.n_in_test, self.n_out_test)

		rois = loc2bbox(anchors, locs)
		rois[:, 0::2].clamp_(0, img_size[1])
		rois[:, 1::2].clamp_(0, img_size[0])

		min_size = self.min_size
		roi_w = rois[:, 2] - rois[:, 0]
		roi_h = rois[:, 3] - rois[:, 1]
		keep_index = torch.where((roi_h >= min_size) & (roi_w >= min_size))[0]
		if len(keep_index) == 0:
			keep_index = [0]
		rois = rois[keep_index]
		scores = scores[keep_index]

		keep_index = scores.argsort(descending=True)[:n_in]
		rois = rois[keep_index]
		scores = scores[keep_index]

		keep_index = nms(rois, scores, self.nms_threshold)
		rois = rois[keep_index]

		return rois[:n_out]


class RPNTargetGenerator:
	"""generate ground truth for RPN"""
	def __init__(self, n_sample=256, iou_threshold_pos=0.7, iou_threshold_neg=0.3, pos_ratio=0.5):
		self.n_sample = n_sample
		self.iou_threshold_pos = iou_threshold_pos
		self.iou_threshold_neg = iou_threshold_neg
		self.pos_ratio = pos_ratio

	def __call__(self, anchors, bbox, img_size, vis=None):
		"""
		Args:
			anchors: (R, 4)
			bbox: (B, 4)
			img_size: (2)
		"""
		target_loc = torch.zeros((len(anchors), 4)).cuda()
		target_label = -torch.ones(len(anchors), dtype=torch.long).cuda()
		inside_index = torch.where(
			(anchors[:, 0] >= 0) &
			(anchors[:, 1] >= 0) &
			(anchors[:, 2] <= img_size[1]) &
			(anchors[:, 3] <= img_size[0])
		)[0]
		anchors = anchors[inside_index]
		argmax_iou_over_bbox, labels = self._generate_label(anchors, bbox)
		loc = bbox2loc(anchors, bbox[argmax_iou_over_bbox])
		target_loc[inside_index] = loc
		target_label[inside_index] = labels

		return target_loc, target_label

	def _generate_label(self, anchor, bbox):
		label = -torch.ones(len(anchor), dtype=torch.long).cuda()
		ious = box_iou(anchor, bbox)
		max_iou_over_anchor, _ = ious.max(0)
		max_iou_over_bbox, argmax_iou_over_bbox = ious.max(1)
		argmax_iou_over_anchor = torch.where(ious == max_iou_over_anchor)[0]

		label[max_iou_over_bbox <= self.iou_threshold_neg] = 0
		label[argmax_iou_over_anchor] = 1
		label[max_iou_over_bbox >= self.iou_threshold_pos] = 1

		n_pos = int(self.n_sample * self.pos_ratio)
		pos_index = torch.where(label == 1)[0].cpu().numpy()
		if len(pos_index) > n_pos:
			remove_index = np.random.choice(pos_index, size=len(pos_index) - n_pos, replace=False)
			label[remove_index] = -1
		n_neg = self.n_sample - (label == 1).sum().cpu().numpy()
		neg_index = torch.where(label == 0)[0].cpu().numpy()
		if len(neg_index) > n_neg:
			remove_index = np.random.choice(neg_index, size=len(neg_index) - n_neg, replace=False)
			label[remove_index] = -1

		return argmax_iou_over_bbox, label


class RoITargetGenerator:
	def __init__(self,
				 n_samples=128,
				 pos_ratio=0.25,
				 iou_threshold_pos=0.5,
				 iou_threshold_neg_hi=0.5,
				 iou_threshold_neg_lo=0.1):
		self.n_samples = n_samples
		self.pos_ratio = pos_ratio
		self.iou_threshold_pos = iou_threshold_pos
		self.iou_threshold_neg_hi = iou_threshold_neg_hi
		self.iou_threshold_neg_lo = iou_threshold_neg_lo

	def __call__(self, roi, bbox, label, loc_normalize_mean, loc_normalize_std):
		"""
		Args:
			roi: (R, 4)
			bbox: (B, 4)
			label: (B)
		"""
		roi = torch.cat([roi, bbox], 0)
		ious = box_iou(roi, bbox)
		max_iou_over_bbox, argmax_iou_over_bbox = ious.max(1)
		gt_roi_label = label[argmax_iou_over_bbox]

		n_pos = int(self.n_samples * self.pos_ratio)
		pos_index = torch.where(max_iou_over_bbox >= self.iou_threshold_pos)[0].cpu()
		n_pos = min(n_pos, len(pos_index))
		pos_index = np.random.choice(pos_index, size=n_pos, replace=False)
		pos_index = torch.from_numpy(pos_index)

		neg_index = torch.where((self.iou_threshold_neg_lo <= max_iou_over_bbox) &
								(max_iou_over_bbox < self.iou_threshold_neg_hi))[0].cpu()
		n_neg = min(self.n_samples - n_pos, len(neg_index))
		neg_index = np.random.choice(neg_index, size=n_neg, replace=False)
		neg_index = torch.from_numpy(neg_index)

		keep_index = torch.cat([pos_index, neg_index], 0)
		sample_roi = roi[keep_index]
		gt_roi_label = gt_roi_label[keep_index]
		gt_roi_label[n_pos:] = 0

		gt_roi_loc = bbox2loc(sample_roi, bbox[argmax_iou_over_bbox[keep_index]])
		gt_roi_loc = (gt_roi_loc - loc_normalize_mean) / loc_normalize_std

		return sample_roi, gt_roi_loc, gt_roi_label
