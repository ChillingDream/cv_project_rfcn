import torch
import numpy as np


def bbox2loc(bbox_a, bbox):
	""" compute offsets and scales given target and source bounding boxes.

	Args:
		bbox: (R, 4) contains (x1, y1, x2, y2)
		bbox_a: (R, 4) contains (x1, y1, x2, y2)

	Returns:
		(R, 4) contains (tx, ty, tw, th)
	"""
	assert bbox.size(1) == bbox_a.size(1) == 4 and bbox.size(0) == bbox_a.size(0)

	w_a = bbox_a[:, 2] - bbox_a[:, 0]
	h_a = bbox_a[:, 3] - bbox_a[:, 1]
	ctr_x_a = bbox_a[:, 0] + 0.5 * w_a
	ctr_y_a = bbox_a[:, 1] + 0.5 * h_a

	w = bbox[:, 2] - bbox[:, 0]
	h = bbox[:, 3] - bbox[:, 1]
	ctr_x = bbox[:, 0] + 0.5 * w
	ctr_y = bbox[:, 1] + 0.5 * h

	tx = (ctr_x - ctr_x_a) / w_a
	ty = (ctr_y - ctr_y_a) / h_a
	th = torch.log(h / h_a)
	tw = torch.log(w / w_a)

	return torch.stack([tx, ty, tw, th]).transpose(0, 1)


def loc2bbox(bbox_a, loc):
	""" compute bounding boxes from offsets and scales

	Args:
		bbox_a: (R, 4) contains (x1, y1, x2, y2)
		loc: (R, 4) contains (tx, ty, tw, th)

	Returns:
		(R, 4)
	"""
	assert bbox_a.size(1) == loc.size(1) == 4 and bbox_a.size(0) == loc.size(0)

	w_a = bbox_a[:, 2] - bbox_a[:, 0]
	h_a = bbox_a[:, 3] - bbox_a[:, 1]
	ctr_x_a = bbox_a[:, 0] + 0.5 * w_a
	ctr_y_a = bbox_a[:, 1] + 0.5 * h_a

	w = w_a * torch.exp(loc[:, 2])
	h = h_a * torch.exp(loc[:, 3])
	ctr_x = w_a * loc[:, 0] + ctr_x_a
	ctr_y = h_a * loc[:, 1] + ctr_y_a

	bbox = torch.zeros_like(bbox_a)
	bbox[:, 0] = ctr_x - 0.5 * w
	bbox[:, 1] = ctr_y - 0.5 * h
	bbox[:, 2] = ctr_x + 0.5 * w
	bbox[:, 3] = ctr_y + 0.5 * h

	return bbox


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
	anchors = torch.zeros((len(ratios) * len(anchor_scales), 4), dtype=torch.float32)

	for i, ratio in enumerate(ratios):
		for j, scale in enumerate(anchor_scales):
			w = base_size * scale * np.sqrt(ratio)
			h = base_size * scale / np.sqrt(ratio)

			k = i * len(anchor_scales) + j
			anchors[k, 0] = -w / 2
			anchors[k, 1] = -h / 2
			anchors[k, 2] = w / 2
			anchors[k, 3] = h / 2

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
	shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
	shift = torch.stack([shift_x, shift_y], dim=2)
	shift = shift.reshape(-1, 2).repeat(1, 2)
	shifted_anchors = anchors + shift.unsqueeze(1)

	return shifted_anchors.reshape(-1, 4).float()

