import torch
import numpy as np 
from BDD10K_dataset import BDD10K_dataset
from Pascal_VOC_dataset import Pascal_VOC_dataset
from torch.utils.data import Dataset, DataLoader, TensorDataset

# bdd_dataset = BDD10K_dataset(bdd10k_path='/home/zkj/codes/cv_project_rfcn/data/bdd100k', dataset_list=['train'], dump_to='bdd10k_train.pkl')
bdd_dataset2 = BDD10K_dataset(bdd10k_path='/home/zkj/codes/cv_project_rfcn/data/bdd100k', dataset_list=['val'], dump_to='bdd10k_val.pkl')
print('bdd10k dump!')