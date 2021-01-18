import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg16
from torchvision.models.resnet import resnet101
from .rpn import RPN
from torchvision.ops import PSRoIPool, nms, RoIPool
from utils import loc2bbox


class RFCN(nn.Module):
	def __init__(self, config, backbone='resnet-101', head='rfcn'):
		super().__init__()
		self.extractor, classifier = self._get_backbone(backbone)
		self.multi_loc = head == 'rcnn'
		if backbone == 'resnet-101':
			self.feat_stride = 32
			self.rpn = RPN(in_channels=1024, mid_channels=512, feat_stride=self.feat_stride)
			if head == 'rfcn':
				self.roi_head = RFCNRoIhead(2048, 1024, 3, config.num_classes, 1 / self.feat_stride)
			elif head == 'rcnn':
				self.roi_head = RCNNRoIhead(
					nn.Sequential(nn.Linear(50176, 512), nn.ReLU(True)),
					512,
					config.num_classes,
					roi_size=7,
					scale=1 / self.feat_stride
				)
		elif backbone == 'vgg16':
			self.feat_stride = 16
			self.rpn = RPN(in_channels=512, mid_channels=512, feat_stride=self.feat_stride)
			if head == 'rfcn':
				raise NotImplementedError
				#self.roi_head = RFCNRoIhead(2048, 1024, 3, config.num_classes, 1 / self.feat_stride)
			elif head == 'rcnn':
				self.roi_head = RCNNRoIhead(
					classifier,
					4096,
					config.num_classes,
					roi_size=7,
					scale=1 / self.feat_stride
				)
		else:
			raise NotImplementedError

		self.loc_normalize_mean = torch.tensor([0., 0., 0., 0.]).cuda()
		self.loc_normalize_std = torch.tensor([.1, .1, .2, .2]).cuda()
		self.nms_threshold = None
		self.score_threshold = None
		self.n_class = config.num_classes
		self._set_threshold('eval')

	def forward(self, x, vis=False):
		"""
		Args:
			x: (N, C, H, W)
		"""
		img_size = x.size()[2:]
		h = self.extractor(x)
		rpn_scores, rpn_locs, rois, roi_indices, _ = self.rpn(h, img_size, vis)
		roi_scores, roi_locs = self.roi_head(h, rois, roi_indices)

		return roi_scores, roi_locs, rois, roi_indices

	def predict(self, imgs, scale=1, vis=False):
		if vis:
			self._set_threshold('vis')
		else:
			self._set_threshold('eval')

		bboxes, labels, scores = [], [], []
		self.eval()
		with torch.no_grad():
			for img in imgs:
				img_size = img.size()[1:]
				roi_score, roi_loc, rois, _ = self.forward(img.unsqueeze(0), vis)

				mean, std = self.loc_normalize_mean, self.loc_normalize_std
				if self.multi_loc:
					roi_loc = roi_loc.view(-1, self.n_class, 4)
					rois = rois.view(-1, 1, 4).expand_as(roi_loc)

				roi_loc = roi_loc * std + mean
				roi_bbox = loc2bbox(rois.reshape(-1, 4), roi_loc.reshape(-1, 4))
				roi_bbox[:, 0::2].clamp_(0, img_size[1])
				roi_bbox[:, 1::3].clamp_(0, img_size[0])
				cls_score = F.softmax(roi_score, dim=1)

				if self.multi_loc:
					roi_bbox = roi_bbox.reshape((-1, self.n_class, 4))
					for c in range(1, self.n_class):
						bbox, score = self._pred_nms(roi_bbox[:, c, :], cls_score[:, c])
						bboxes.append(bbox)
						scores.append(score)
						labels.append(torch.ones(len(bbox), dtype=torch.long) * c)

					bboxes = torch.cat(bboxes, 0)
					scores = torch.cat(scores, 0)
					labels = torch.cat(labels, 0)

		self.train()
		return bboxes, labels, scores

	def _pred_nms(self, roi_bbox, score):
		keep_index = score > self.score_threshold
		bbox, score = roi_bbox[keep_index], score[keep_index]
		keep_index = nms(bbox, score, self.nms_threshold)
		bbox, score = bbox[keep_index], score[keep_index]

		return bbox, score

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
				resnet.layer4,
				nn.Conv2d(2048, 1024, 1, 1, 0)
			)
			return extractor, None
		elif backbone == 'vgg16':
			vgg = vgg16(True)
			features = list(vgg.features)[:30]
			classifier = list(vgg.classifier)
			del classifier[6]
			del classifier[5]
			del classifier[2]
			classifier = nn.Sequential(*classifier)
			for layer in features[:10]:
				for p in layer.parameters():
					p.requires_grad = False
			features = nn.Sequential(*features)
			return features, classifier
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
		self.psroi_pool = PSRoIPool((bin_size, bin_size), scale)

		nn.init.normal_(self.conv1.weight, 0, 0.01)
		nn.init.normal_(self.conv_cls.weight, 0, 0.01)
		nn.init.normal_(self.conv_loc.weight, 0, 0.01)

	def forward(self, h, rois, roi_indices):
		h = self.conv1(h)
		h_cls = self.conv_cls(h)
		h_reg = self.conv_loc(h)
		roi_indices = roi_indices.unsqueeze(1).float()
		indices_and_rois = torch.cat([roi_indices, rois], 1).float()
		roi_score = self.psroi_pool(h_cls, indices_and_rois).mean(dim=[2, 3])
		roi_locs = self.psroi_pool(h_reg, indices_and_rois).mean(dim=[2, 3])
		return roi_score, roi_locs


class RCNNRoIhead(nn.Module):
	def __init__(self, classifier, mid_channels, num_classes, roi_size, scale):
		super().__init__()
		self.classifier = classifier
		self.cls = nn.Linear(mid_channels, num_classes)
		self.loc = nn.Linear(mid_channels, num_classes * 4)
		self.roi_pool = RoIPool((roi_size, roi_size), scale)

		nn.init.normal_(self.cls.weight, 0, 0.01)
		nn.init.normal_(self.loc.weight, 0, 0.01)

	def forward(self, h, rois, roi_indices):
		roi_indices = roi_indices.unsqueeze(1).float()
		indices_and_rois = torch.cat([roi_indices, rois], 1).float()
		h = self.roi_pool(h, indices_and_rois)
		h = h.flatten(start_dim=1)
		h = self.classifier(h)
		roi_score = self.cls(h)
		roi_locs = self.loc(h)
		return roi_score, roi_locs

