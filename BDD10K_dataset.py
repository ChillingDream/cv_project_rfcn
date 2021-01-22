import os
import json
import random
import pickle
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm, trange
from skimage import transform as sktsf
from torch.utils.data import Dataset
from torchvision import transforms as tvtsf


class BDD10K_dataset(Dataset):
    def __init__(self, bdd10k_path=None, dataset_list=['train'], min_size=600, max_size=1000, max_objs=-1, dump_to=None, load_from=None):

        self._classes = ("bike",
                        "bus",
                        "car",
                        "motor",
                        "person",
                        "rider",
                        "traffic light",
                        "traffic sign",
                        "train",
                        "truck")
        self._num_classes = len(self._classes)
        self._class_to_ind = dict(zip(self._classes, range(self._num_classes)))
        self.min_size = min_size
        self.max_size = max_size
        self.max_objs = max_objs

        if load_from:
            print('loading data...')
            with open(load_from, 'rb') as fin:
                self._dataset = pickle.load(fin)
        else:
            if not bdd10k_path:
                print('wrong! bdd10k_path missing')
                exit(0)
            self._dataset = []
            self._bdd100k_path = bdd10k_path
            # self._labels_paths = [os.path.join(self._bdd100k_path, 'labels', 'bdd100k_labels_images_%s.json' % dataset) for dataset in dataset_list]
            self._labels_paths = [os.path.join(self._bdd100k_path, 'labels', 'bdd100k_labels_images_%s.json' % 'train')]
            self._images_paths = [os.path.join(self._bdd100k_path, 'images', '10k', dataset) for dataset in dataset_list]
            # self._images_paths = [os.path.join(self._bdd100k_path, 'images', '10k', 'train') ]
            # print(self._images_paths)
            # self.annos = dict((dataset_list[ind] ,self._load_BDD100K_annotation(label_path)) for ind, label_path in enumerate(self._labels_paths))
            self._load_all_data()
            if dump_to:
                with open(dump_to, 'wb') as fout:
                    pickle.dump(self._dataset, fout)


    def __len__(self):
        return len(self._dataset)

    def _load_BDD100K_annotation(self, Annotations_file, min_box_size=32):   
        filename = Annotations_file
        # print(filename)
        if not os.path.exists(filename):
            return None
        # print(filename)
        # print('parsing json...')
        with open(filename) as f:
            annos = json.load(f)
        annos_list = []
        for anno in annos:
            boxes = []
            gt_classes = []
            for obj in anno['labels']:
                # print(obj['category'].lower().strip())
                if obj['category'].lower().strip() not in self._classes:
                    continue
                # print('obj find!!!')
                cls = self._class_to_ind[obj['category'].lower().strip()]
                
                x1 = float(obj["box2d"]["x1"]) - 1
                y1 = float(obj["box2d"]["y1"]) - 1
                x2 = float(obj["box2d"]["x2"]) - 1
                y2 = float(obj["box2d"]["y2"]) - 1
                if min_box_size > 0 and np.abs(x1 - x2) * np.abs(y1 - y2) < min_box_size:
                    continue
                boxes.append([x1, y1, x2, y2])
                gt_classes.append(cls)
                if self.max_objs > 0 and len(boxes) >= self.max_objs:
                    break
            if not boxes or not gt_classes:
                continue
            annos_list.append({"name": anno["name"], 'boxes': np.array(boxes, dtype=np.float32), 'gt_classes': np.array(gt_classes, dtype=np.long)})
        # print('anno loading done')
        return annos_list

    def _prep_im_for_blob(self, im, pixel_means, target_size, max_size):
        """Mean subtract and scale an image for use in a blob."""
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

        return im, im_scale

    def _im_list_to_blob(self, ims):
        """Convert a list of images into a network input.
        Assumes images are already prepared (means subtracted, BGR order, ...).
        """
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        num_images = len(ims)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                        dtype=np.float32)
        for i in range(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

        return blob

    def _preprocess(self, img, min_size, max_size, bbox=None, flipped=False):
        if img.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            img = img[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W)
            img = img.transpose((2, 0, 1))
        if flipped:
            img = img[:, ::-1, :]
        img = np.array(img).astype('float32')
        # print('img shape:', img.shape)
        C, H, W = img.shape
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        img = img / 255.
        scale = min(scale1, scale2)
        img = sktsf.resize(
            img,
            (C, H * scale, W * scale),
            mode='reflect',
            anti_aliasing=False
        )
        normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        img = normalize(torch.from_numpy(img))
        C, o_H, o_W = img.shape
        if bbox is not None:
            bbox = self._resize_bbox(bbox, (H, W), (o_H, o_W))
        return img.numpy(), bbox, scale

    @staticmethod
    def inverse_normalize(img):
        return ((img * np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1)) +
                 np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))).clip(min=0, max=1) * 255).astype(np.uint8)

    def _resize_bbox(self, bbox, in_size, out_size):
        y_scale = float(out_size[0]) / in_size[0]
        x_scale = float(out_size[1]) / in_size[1]
        bbox[:, 0] = x_scale * bbox[:, 0]
        bbox[:, 2] = x_scale * bbox[:, 2]
        bbox[:, 1] = y_scale * bbox[:, 1]
        bbox[:, 3] = y_scale * bbox[:, 3]
        return bbox

    def _flip_bbox(self, bbox, size, y_flip=False, x_flip=False):
        H, W = size
        bbox = bbox.copy()
        if y_flip:
            y_max = H - bbox[:, 1]
            y_min = H - bbox[:, 3]
            bbox[:, 1] = y_min
            bbox[:, 3] = y_max
        if x_flip:
            x_max = W - bbox[:, 0]
            x_min = W - bbox[:, 2]
            bbox[:, 0] = x_min
            bbox[:, 2] = x_max
        return bbox

    def _random_flip(self, img, y_random=False, x_random=False,
                     return_param=False, copy=False):
        y_flip, x_flip = False, False
        if y_random:
            y_flip = random.choice([True, False])
        if x_random:
            x_flip = random.choice([True, False])

        if y_flip:
            img = img[:, ::-1, :]
        if x_flip:
            img = img[:, :, ::-1]

        if copy:
            img = img.copy()

        if return_param:
            return img, {'y_flip': y_flip, 'x_flip': x_flip}
        else:
            return img

    def _transform(self, in_data):
        img, bbox, label = in_data
        img, bbox, scale = self._preprocess(img, self.min_size, self.max_size, bbox)
        _, o_H, o_W = img.shape

        # horizontally flip
        img, params = self._random_flip(
            img, x_random=True, return_param=True)
        bbox = self._flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale

    def _load_all_data(self):
        for ind, label_path in enumerate(self._labels_paths):
            print('loading annos...')
            annos = self._load_BDD100K_annotation(label_path)
            images_path = self._images_paths[ind]
            for anno in tqdm(annos):
                if not anno:
                    continue
                # print('find anno!')
                image_file = os.path.join(images_path, anno['name'])
                if not os.path.exists(image_file):
                    continue
                print('find image!')
                f = Image.open(image_file)
                try:
                    img = f.convert('RGB')
                    img = np.asarray(img, dtype=np.float32)
                finally:
                    if hasattr(f, 'close'):
                        f.close()
                img, bbox, label, scale = self._transform((img, anno['boxes'], anno['gt_classes']))
                self._dataset.append((img, bbox, label, scale))

    def __getitem__(self, i):
        img, bbox, label, scale = self._dataset[i]
        return img.copy(), bbox.copy(), label.astype(np.int64).copy(), scale
