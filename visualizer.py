import torch
import visdom
import numpy as np
from matplotlib import pyplot as plt


def vis_image(img, ax=None):
	if not ax:
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
	ax.imshow(img.astype(np.int8))
	return ax


def vis_bbox(img, bboxes, label=None, score=None, ax=None):
	ax = vis_image(img, ax)
	for i, bbox in enumerate(bboxes):
		xy = bbox[1], bbox[0]
		height = bbox[2] - bbox[0]
		width = bbox[3] - bbox[1]
		ax.add_patch(plt.Rectangle(xy, width, height, fill=False, edgecolor='red', linewidth=2))
		caption = []
		if label:
			caption.append("%d" % label[i])
		if score:
			caption.append("%.2f" % score[i])
		if caption:
			ax.text(xy[0], xy[1], ': '.join(caption), style='italic', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
	return ax


class Visualizer:
	def __init__(self, env='default'):
		self.vis = visdom.Visdom('localhost', env=env, use_incoming_socket=False)
		self.index = {}

	def plot(self, name, y, **kwargs):
		x = self.index.get(name, 0)
		self.vis.line(Y=[y], X=[x], win=name, opts={"title": name}, update='append', **kwargs)
		self.index[name] = x + 1

	def multi_plot(self, kv):
		for k, v in kv.items():
			if v:
				self.plot(k, v)

	def show_image(self, name, imgs):
		self.vis.images(torch.tensor(imgs), win=name, opts={'title': name})

	def show_multi_image(self, kv):
		for k, v in kv:
			if v:
				self.plot(k, v)

	def show_image_bbox(self, name, img, bbox, label, score):
		self.show_image(name, vis_bbox(img, bbox, label, score))

	def log(self, info):
		self.vis.text(info, 'log_text')

	def __getattr__(self, item):
		return getattr(self.vis, item)
