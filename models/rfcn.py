import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet101
from .rpn import RPN
from torchvision.ops import PSRoIPool, nms
from utils import loc2bbox


class RFCN(nn.Module):
	def __init__(self, config, backbone='resnet-101'):
		super().__init__()
		self.extractor = self._get_backbone(backbone)
		if backbone == 'resnet-101':
			self.feat_stride = 32
			self.rpn = RPN(in_channels=2048, mid_channels=512, feat_stride=self.feat_stride)
			self.RoIhead = RFCNRoIhead(2048, 1024, 3, config.num_classes, 1 / self.feat_stride)
		else:
			raise NotImplementedError
		self.loc_normalize_mean = torch.tensor([0., 0., 0., 0.]).cuda()
		self.loc_normalize_std = torch.tensor([.1, .1, .2, .2]).cuda()
		self.nms_threshold = None
		self.score_threshold = None
		self._set_threshold('eval')

	def forward(self, x):
		"""
		Args:
			x: (N, C, H, W)
		"""
		img_size = x.size()[2:]
		h = self.extractor(x)
		rpn_scores, rpn_locs, rois, roi_indices, _ = self.rpn(h, img_size)
		roi_scores, roi_locs = self.RoIhead(h, rois, roi_indices)

		return roi_scores, roi_locs, rois, roi_indices

	def predict(self, imgs, scale=1, vis=False):
		if vis:
			self._set_threshold('vis')
		else:
			self._set_threshold('eval')

		bboxes = []
		labels = []
		scores = []
		self.eval()
		with torch.no_grad():
			for img in imgs:
				img_size = img.size()[1:]
				roi_score, roi_loc, rois, _ = self.forward(img.unsqueeze(0))
				mean, std = self.loc_normalize_mean, self.loc_normalize_std
				roi_loc = roi_loc * std + mean
				roi_bbox = loc2bbox(rois, roi_loc)
				roi_bbox[:, 0::2].clamp_(0, img_size[1])
				roi_bbox[:, 1::3].clamp_(0, img_size[0])
				score, label = F.softmax(roi_score, dim=1).max(1)

				keep_index = score > self.score_threshold
				bbox, score, label = roi_bbox[keep_index], score[keep_index], label[keep_index]
				keep_index = nms(bbox, score, self.nms_threshold)
				bbox, score, label = roi_bbox[keep_index], score[keep_index], label[keep_index]
				bboxes.append(bbox)
				scores.append(score)
				labels.append(label)

		self.train()
		return bboxes, labels, scores

	def _get_backbone(self, backbone):
		if backbone == 'resnet-101':
			resnet = resnet101(pretrained=True)
			extractor = nn.Sequential(
				resnet.conv1,
				resnet.bn1,
				resnet.relu,
				resnet.maxpool,
				resnet.layer1,
				resnet.layer2,
				resnet.layer3,
				resnet.layer4
			)
			return extractor
		else:
			raise NotImplementedError

	def _set_threshold(self, mode):
		if mode == 'eval':
			self.nms_threshold = 0.3
			self.score_threshold = 0.05
		elif mode == 'vis':
			self.nms_threshold = 0.3
			self.score_threshold = 0.7
		else:
			raise ValueError('no such mode.')


class RFCNRoIhead(nn.Module):
	def __init__(self, in_channels, mid_channels, bin_size, num_classes, scale):
		super().__init__()
		self.bin_size = bin_size
		self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0)
		self.conv_cls = nn.Conv2d(mid_channels, bin_size ** 2 * num_classes, 1, 1, 0)
		self.conv_loc = nn.Conv2d(mid_channels, bin_size ** 2 * 4, 1, 1, 0)
		self.RoIpool = PSRoIPool((bin_size, bin_size), scale)

	def forward(self, h, rois, roi_indices):
		h = self.conv1(h)
		h_cls = self.conv_cls(h)
		h_reg = self.conv_loc(h)
		roi_indices = roi_indices.unsqueeze(1).float()
		indices_and_rois = torch.cat([roi_indices, rois], 1).float()
		roi_score = self.RoIpool(h_cls, indices_and_rois).mean(dim=[2, 3])
		roi_locs = self.RoIpool(h_reg, indices_and_rois).mean(dim=[2, 3])
		return roi_score, roi_locs

