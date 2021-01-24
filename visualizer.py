import matplotlib
import numpy as np
import visdom

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def fig2visdom(fig):
	"""convert a matplotlib figure to visdom data"""
	fig = fig.get_figure()
	fig.canvas.draw()
	w, h = fig.canvas.get_width_height()
	buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
	buf.shape = (w, h, 4)
	buf = np.roll(buf, 3, axis=2)
	buf = buf.reshape((h, w, 4))
	plt.close()
	return buf[:, :, :3].transpose((2, 0, 1)) / 255


def vis_image(img, ax=None):
	if not ax:
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
	ax.imshow(img.astype(np.uint8))
	return ax


def vis_bbox(img, bboxes, label=None, score=None, ax=None):
	ax = vis_image(img, ax)
	for i, bbox in enumerate(bboxes):
		xy = bbox[0], bbox[1]
		width = bbox[2] - bbox[0]
		height = bbox[3] - bbox[1]
		ax.add_patch(plt.Rectangle(xy, width, height, fill=False, edgecolor='red', linewidth=2))
		caption = []
		if label is not None:
			caption.append("%d" % label[i])
		if score is not None:
			caption.append("%.2f" % score[i])
		if caption:
			ax.text(xy[0], xy[1], ': '.join(caption), style='italic',
					bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
	return fig2visdom(ax)


class Visualizer:
	def __init__(self, env='default'):
		self.vis = visdom.Visdom('localhost', env=env, use_incoming_socket=False, port=8907)
		self.index = {}

	def plot(self, name, y, **kwargs):
		x = self.index.get(name, 0)
		self.vis.line(Y=[y], X=[x], win=name, opts={"title": name}, update='append', **kwargs)
		self.index[name] = x + 1

	def multi_plot(self, kv):
		for k, v in kv.items():
			if v:
				self.plot(k, v)

	def show_image(self, name, img):
		assert np.min(img) >= 0
		self.vis.images(img, win=name, opts={'title': name})

	def show_multi_image(self, kv):
		for k, v in kv:
			if v:
				self.plot(k, v)

	def show_image_bbox(self, name, img, bbox, label=None, score=None):
		img = img.transpose(1, 2, 0)
		self.show_image(name, vis_bbox(img, bbox, label, score))

	def log(self, info):
		self.vis.text(info, 'log_text')

	def __getattr__(self, item):
		return getattr(self.vis, item)
