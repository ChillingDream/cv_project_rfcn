# Vehicle detection based on Faster R-CNN
### Requrements
* torch>=1.4.0
* torchvision>=0.5.0

othres:

`pip install -r requirements`
### Prepare data
### VOC
Download data
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
Extract data
```bash
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCtrainval_11-May-2012.tar
```
Under project root directory, there should be a folder `VOCdevkit` which contains two folders `VOC2007` and `VOC2012`. If you fail to download from given urls, you download in other way and put organize them in that structure.

### BDD

Data can be found on the Internet.
Website of BDD100K:https://bdd-data.berkeley.edu/ 

### Prepare the BDD dataset

It is recommended to preprocess the data set before using the BDD100K dataset. 

BDD 100K preprocessing process:
1. run `python BDD_100K_preprocessing.py -b {bdd100k_path} -t {train_dump_path} -v {val_dump_path}`
2. run `python train.py BDD -t {train_dump_path} -v {val_dump_path}`

Preprocessing may take a few hours.

### Setup Visdom server
`python -m visdom.server &`

This is for visualization on `localhost:8097`.

### Train Pascal VOC
`python train.py VOC`
#### Single category training
Set parameter `just_car` to True when creating a `Pascal_VOC_dataset` or `BDD100K_dataset` object.

### Test
`python test.py [VOC|BDD]`