# Multi-Label Semantics Preserving based Deep Cross-Modal Hashing (MLSPH)

Pytorch implementation of paper 'Multi-Label Semantics Preserving based Deep Cross-Modal Hashing'.

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
journal = {Signal Processing: Image Communication},   
doi = {}   
}  

---
### Dependencies 
you need to install these package to run
- visdom 0.1.8+
- pytorch 1.0.0+
- tqdm 4.0+  
- python 3.5+
----

### Dataset

we implement our method on dataset Mirflckr25K:

(1) please download the original image-text data of Mirflckr25K from http://press.liacs.nl/mirflickr/mirdownload.html  and put it under the folder /dataset/data/.

(2) please download the mirflickr25k-fall.mat, mirflickr25k-iall.mat, mirflickr25k-lall.mat and mirflickr25k-yall.mat from https://pan.baidu.com/s/1FX82NhdtnTeARcgmqxYCag 
(提取码：imk4) and put them under the folder /dataset/data/.

### How to run
 
 Step1: Run your Visdom: python -m visdom.server
 
 Step2: Run function run in main.py

If you have any problems, please feel free to contact Xitao Zou (xitaozou@mail.swu.edu.cn).
