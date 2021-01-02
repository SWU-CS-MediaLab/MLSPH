# Multi-Label Semantics Preserving based Deep Cross-Modal Hashing (MLSPH)
## Abstract

Due to the storage and retrieval efficiency of hashing, as well as the highly discriminative feature extraction by deep neural networks, deep cross-modal hashing retrieval has been attracting increasing attention in recent years. However, most of existing deep cross-modal hashing methods simply employ single-label to directly measure the semantic relevance across different modalities, but neglect the potential contributions from multiple category labels. With the aim to improve the accuracy of cross-modal hashing retrieval by fully exploring the semantic relevance based on multiple labels of training data, in this paper, we propose a multi-label semantics preserving based deep cross-modal hashing (MLSPH) method. MLSPH firstly utilizes multi-labels of instances to calculate semantic similarity of the original data. Subsequently, a memory bank mechanism is introduced to preserve the multiple labels semantic similarity constraints and enforce the distinctiveness of learned hash representations over the whole training batch. Extensive experiments on several benchmark datasets reveal that the proposed MLSPH surpasses prominent baselines and reaches the state-of-the-art performance in the field of cross-modal hashing retrieval. 

------

Please cite our paper if you use this code in your own work:
@article{zou2021,  
author = {Zou, Xitao and Wang, Xinzhi and Bakker, Erwin and Wu, Song},    
year = {2021},    
month = {01},    
pages = {},  
title = {Multi-Label Semantics Preserving based Deep Cross-Modal Hashing},    
journal = {signal processing image communication},   
doi = {}   
}  

---
### Dependencies 
you need to install these package to run
- visdom 0.1.8+
- pytorch 1.0.0+
- tqdm 4.0+  
- python 3.5+

---
### Dependencies 
you need to install these package to run
- visdom 0.1.8+
- pytorch 1.0.0+
- tqdm 4.0+  
- python 3.5+
----
### Logs and checkpoints

The training will create a log and checkpoint to store the model. \
You can find log in ./logs/\{method_name\}/\{dataset_name\}/date.txt \
You can find checkpoints in ./checkpoints/\{method_name\}/\{dataset_name\}/\{bit\}-\{model_name\}.pth

----
### How to using
- create a configuration file as ./script/default_config.yml
```yaml
training:
  # the name of python file in training
  method: SCAHN
  # the data set name, you can choose mirflickr25k, nus wide, ms coco, iapr tc-12
  dataName: Mirflickr25K
  batchSize: 64
  # the bit of hash codes
  bit: 64
  # if true, the program will be run on gpu. Of course, you need install 'cuda' and 'cudnn' better.
  cuda: True
  # the device id you want to use, if you want to multi gpu, you can use [id1, id2]
  device: 0
datasetPath:
  Mirflickr25k:
    # the path you download the image of data set. Attention: image files, not mat file.
    img_dir: \dataset\mirflickr25k\mirflickr

```
- run ./script/main.py and input configuration file path.
```python
from torchcmh.run import run
if __name__ == '__main__':
    run(config_path='default_config.yml')
```
- data visualization
Before you start trainer, please use command as follows to open visdom server.
```shell script
python -m visdom.server
```
Then you can see the charts in browser in special port.
