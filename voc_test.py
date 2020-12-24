import torch
import numpy as np 
from Pascal_VOC_dataset import Pascal_VOC_dataset
from torch.utils.data import Dataset, DataLoader, TensorDataset

voc_dataset = Pascal_VOC_dataset(devkit_path = 'VOCdevkit', dataset_list = ['2007_trainval','2012_trainval'], load_all=True)
for i in range(0):
    img, bbox, label, scale = voc_dataset[i]
    #print(img)
    print(bbox)
    print(label)
    print(scale)

train_loader = DataLoader(dataset=voc_dataset,
                          batch_size=1,
                          shuffle=True)

for i, data in enumerate(train_loader):
    if i == 1000:
        break
    print(len(data[2]))