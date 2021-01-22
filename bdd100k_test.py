import torch
import numpy as np 
from BDD100K_dataset import BDD100K_dataset
from Pascal_VOC_dataset import Pascal_VOC_dataset
from torch.utils.data import Dataset, DataLoader, TensorDataset

bdd_dataset = BDD100K_dataset(bdd100k_path='/home/zkj/codes/cv_project_rfcn/data/bdd100k', dataset_list=['train'], dump_to='bdd100k_small_train.pkl')
bdd_dataset2 = BDD100K_dataset(bdd100k_path='/home/zkj/codes/cv_project_rfcn/data/bdd100k', dataset_list=['val'], dump_to='bdd100k_small_val.pkl')
print('bdd100k dump!')
# bdd_dataset3 = BDD100K_dataset(bdd100k_path='/home/zkj/codes/cv_project_rfcn/data/bdd100k', dataset_list=['train'], dump_to='bdd100k_train.pkl')
# bdd_dataset4 = BDD100K_dataset(bdd100k_path='/home/zkj/codes/cv_project_rfcn/data/bdd100k', dataset_list=['train'], dump_to='bdd100k_train.pkl')

# bdd_dataset = BDD100K_dataset(bdd100k_path='/home/zkj/codes/cv_project_rfcn/data/bdd100k', dataset_list=['train'], load_from='bdd100k_train.pkl')

# for i in range(10):
#     img, bbox, label, scale = bdd_dataset[i]
#     #print(img)
#     print(bbox)
#     print(label)
#     print(scale)

# train_loader = DataLoader(dataset=bdd_dataset,
#                           batch_size=1,
#                           shuffle=True)

# for i, data in enumerate(train_loader):
#     if i == 1000:
#         break
#     print(len(data[2]))