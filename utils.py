import torch
import numpy as np


def bbox2loc(bbox_G, bbox_P):
	""" compute offsets and scales given target and source bounding boxes.

	Args:
		bbox_G: (R, 4) contains (y1, x1, y2, x2)
		bbox_P: (R, 4) contains (y1, x1, y2, x2)

	Returns:
		(R, 4) contains (ty, tx, th, tw)
	"""
	assert bbox_G.size(1) == bbox_P.size(1) == 4 and bbox_G.size(0) == bbox_P.size(0)

	Ph = (bbox_P[:, 2] - bbox_P[:, 0])
	Pw = (bbox_P[:, 3] - bbox_P[:, 1])
	ctr_Py = bbox_P[:, 0] + 0.5 * Ph
	ctr_Px = bbox_P[:, 1] + 0.5 * Pw

	Gh = (bbox_G[:, 2] - bbox_G[:, 0])
	Gw = (bbox_G[:, 3] - bbox_G[:, 1])
	ctr_Gy = bbox_G[:, 0] + 0.5 * Gh
	ctr_Gx = bbox_G[:, 1] + 0.5 * Gw

	tx = (ctr_Gx - ctr_Px) / Pw
	ty = (ctr_Gy - ctr_Py) / Ph
	th = torch.log(Gh / Ph)
	tw = torch.log(Gw / Pw)

	return torch.stack([ty, tx, th, tw]).transpose(0, 1)


def loc2bbox(bbox_P, loc):
	""" compute bounding boxes from offsets and scales

	Args:
		bbox_P: (R, 4) contains (y1, x1, y2, x2)
		loc: (R, 4) contains (ty, tx, th, tw)

	Returns:
		(R, 4)
	"""
	assert bbox_P.size(1) == loc.size(1) == 4 and bbox_P.size(0) == loc.size(0)

	Ph = (bbox_P[:, 2] - bbox_P[:, 0])
	Pw = (bbox_P[:, 3] - bbox_P[:, 1])
	ctr_Py = bbox_P[:, 0] + 0.5 * Ph
	ctr_Px = bbox_P[:, 1] + 0.5 * Pw

	Gh = Ph * torch.exp(loc[:, 2])
	Gw = Pw * torch.exp(loc[:, 3])
	ctr_Gy = Ph * loc[:, 1] + ctr_Py
	ctr_Gx = Pw * loc[:, 0] + ctr_Px

	bbox_G = torch.zeros_like(bbox_P)
	bbox_G[:, 0] = ctr_Gy - 0.5 * Gh
	bbox_G[:, 1] = ctr_Gx - 0.5 * Gw
	bbox_G[:, 2] = ctr_Gy + 0.5 * Gh
	bbox_G[:, 3] = ctr_Gx + 0.5 * Gw

	return bbox_G


def calc_iou(bbox_a, bbox_b):
	"""calculate the iou between two bounding box

	Args:
		bbox_a: (R, 4) contains (y1, x1, y2, x2)
		bbox_b: (R, 4) contains (y1, x1, y2, x2)

	Returns:
		(R, R)
	"""
	assert bbox_a.size(1) == bbox_b.size(1) == 4 and bbox_a.size(0) == bbox_b.size(1)
	i_tl = torch.max(bbox_a[:, None, :2], bbox_b[:, :2])
	i_br = torch.min(bbox_a[:, None, 2:], bbox_b[:, 2:])

	area_i = torch.prod(i_br - i_tl, dim=2) * (i_tl < i_br).all(dim=2)
	area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], dim=1)
	area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], dim=1)

	return area_i / (area_a[:, None] + area_b - area_i)


def generator_anchor(base_size, ratios, anchor_scales):
	"""generator anchors

	Args:
		base_size: (int).
		ratios: (R1).
		anchor_scales: (R2).

	Returns:
		(R1 * R2, 4)
	"""
	ctr_y = base_size / 2
	ctr_x = base_size / 2
	anchors = torch.zeros((len(ratios) * len(anchor_scales), 4), dtype=torch.float32)

	for i, ratio in enumerate(ratios):
		for j, scale in enumerate(anchor_scales):
			h = base_size * scale * np.sqrt(ratio)
			w = base_size * scale / np.sqrt(ratio)

			k = i * len(anchor_scales) + j
			anchors[k, 0] = ctr_y - h / 2
			anchors[k, 1] = ctr_x - w / 2
			anchors[k, 2] = ctr_y + h / 2
			anchors[k, 3] = ctr_x + w / 2

	return anchors


def enumerate_shifted_anchors(anchors, feat_stride, height, width):
	"""apply the anchors to each coordinate of a height * width feature map

	Args:
		anchors: (A, 4)
		feat_stride: (int)
		height: (int)
		width: (int)

	Returns:
		(height * width * A, 4)
	"""
	shift_y = torch.arange(0, height * feat_stride, feat_stride)
	shift_x = torch.arange(0, width * feat_stride, feat_stride)
	shift = torch.stack(torch.meshgrid(shift_y, shift_x), dim=2)
	shift = shift.reshape(-1, 2).repeat(1, 2)
	shifted_anchors = anchors + shift.unsqueeze(1)

	return shifted_anchors.reshape(-1, 4).float()

