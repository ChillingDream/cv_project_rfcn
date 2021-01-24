# cv_project_rfcn
### Requrements
* torch>=1.4.0
* torchvision>=0.5.0

othres:

`pip install -r requirements`
### Prepare the dataset

Dataloader for BDD100K: BDD100K_dataset.py
Dataloader for Pascal VOC: Pascal_VOC_dataset.py

It is recommended to preprocess the data set before using the BDD100K dataset. 

BDD 100K preprocessing process:
1. run `python BDD_100K_preprocessing.py -b {bdd100k_path} -t {train_dump_path} -v {val_dump_path}`
2. rum `python train.py {dataset} -t {train_dump_path} -v {val_dump_path}`

### Train
`python train.py {dataset}`
