import torch
import numpy as np 
from Pascal_VOC_dataset import Pascal_VOC_dataset
from torch.utils.data import Dataset, DataLoader, TensorDataset

voc_dataset = Pascal_VOC_dataset(devkit_path = 'VOCdevkit', dataset_list = ['2007_trainval','2012_trainval'])
for i in range(10):
    img, bbox, label, scale = voc_dataset[i]
    print(img)
    print(bbox)
    print(label)
    print(scale)

train_loader = DataLoader(dataset=voc_dataset,
                          batch_size=32,
                          shuffle=True)

for i, data in enumerate(train_loader):
    if i == 5:
        break
    print(data)