# coding: utf-8
# @Time     : 
# @Author   : Godder
# @Github   : https://github.com/WangGodder

# read image
from PIL import Image
# read mat file
import scipy.io as sio
# use np matrix
import numpy as np
# use torch and torch transform utils
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
# use os to check whether file is exist
import os
from ..base import CrossModalDatasetBase, train_transform, valid_transform

np.random.seed(6)
random_index = np.random.permutation(range(20015))


class DatasetMirflckr25KTrain(CrossModalDatasetBase):
    def __init__(self, img_dir: str, img_mat_url: str, tag_mat_url: str, label_mat_url: str, batch_size=128, train_num=10000, query_num=2000, random_index=random_index):
        super(DatasetMirflckr25KTrain, self).__init__(img_dir, img_mat_url, tag_mat_url, label_mat_url, train_transform, 'Mirflckr25k')

        self.train_num = train_num
        self.batch_size = batch_size
        self.index = random_index[query_num: query_num + train_num]

        img_names = sio.loadmat(img_mat_url)['FAll']    # type: np.ndarray
        img_names = img_names.squeeze()
        self.img_names = img_names[self.index]
        self.img_names = np.array([name[0] for name in self.img_names])
        all_txt = np.array(sio.loadmat(tag_mat_url)['YAll'])
        self.txt = all_txt[self.index]
        all_label = np.array(sio.loadmat(label_mat_url)['LAll'])
        self.label = all_label[self.index]

        self.random_item = []
        self.re_random_item()

    def re_random_item(self):
        self.random_item = []
        for _ in range(self.train_num // self.batch_size):
            random_ind = np.random.permutation(range(self.train_num))
            self.random_item.append(random_ind[:self.batch_size])

    def __len__(self):
        return self.train_num

    def __getitem__(self, item):
        item = self.get_random_item(item)
        img = txt = None
        if self.img_read:
            img = self.read_img(item)
        if self.txt_read:
            txt = torch.Tensor(self.txt[item][np.newaxis, :, np.newaxis])
        label = torch.Tensor(self.label[item])
        index = torch.from_numpy(np.array(item))
        if self.img_read is False:
            return {'index': index, 'txt': txt, 'label': label}
        if self.txt_read is False:
            return {'index': index, 'img': img, 'label': label}

    def get_all_label(self):
        return torch.Tensor(self.label)

    def get_random_item(self, item):
        return self.random_item[item // self.batch_size][item % self.batch_size]


class DatasetMirflckr25KValid(CrossModalDatasetBase):
    def __init__(self, img_dir: str, img_mat_url: str, tag_mat_url: str, label_mat_url: str, step='query', single_model=True, query_num=2000, random_index=random_index):
        super(DatasetMirflckr25KValid, self).__init__(img_dir, img_mat_url, tag_mat_url, label_mat_url, valid_transform, 'Mirflckr25k')
        if step is not 'query' and step is not 'retrieval':
            raise ValueError("step only can be one of 'query' and 'retrieval'!!")
        self.step = step

        self.query_num = query_num
        self.retrieval_num = len(random_index) - query_num
        self.query_index = random_index[:query_num]
        self.retrieval_index = random_index[query_num:]

        img_names = np.array(sio.loadmat(img_mat_url)['FAll'])
        self.query_img_names = img_names[self.query_index]
        self.retrieval_img_names = img_names[self.retrieval_index]
        all_txt = np.array(sio.loadmat(tag_mat_url)['YAll'])
        self.query_txt = all_txt[self.query_index]
        self.retrieval_txt = all_txt[self.retrieval_index]
        all_label = np.array(sio.loadmat(label_mat_url)['LAll'])
        self.query_label = all_label[self.query_index]
        self.retrieval_label = all_label[self.retrieval_index]

    def read_img(self, item):
        if self.step is 'query':
            image_url = os.path.join(self.img_dir, self.query_img_names[item][0][0])
        if self.step is 'retrieval':
            image_url = os.path.join(self.img_dir, self.retrieval_img_names[item][0][0])
        image = Image.open(image_url).convert('RGB')
        image = self.transform(image)
        # image = np.array(image).transpose(2, 0, 1)  # to c, H, W
        # img = torch.Tensor(image)
        return image

    def read_txt(self, item):
        if self.step is 'query':
            return torch.Tensor(self.query_txt[item][np.newaxis, :, np.newaxis])
        if self.step is 'retrieval':
            return torch.Tensor(self.retrieval_txt[item][np.newaxis, :, np.newaxis])

    def read_label(self, item):
        if self.step is 'query':
            return torch.Tensor(self.query_label[item])
        if self.step is 'retrieval':
            return torch.Tensor(self.retrieval_label[item])

    def __len__(self):
        if self.step == 'query':
            return self.query_num
        return self.retrieval_num

    def __getitem__(self, item):
        if self.img_read:
            img = self.read_img(item)
        if self.txt_read:
            txt = self.read_txt(item)
        label = self.read_label(item)
        index = torch.from_numpy(np.array(item))
        if self.img_read is False:
            return {'index': index, 'txt': txt, 'label': label}
        if self.txt_read is False:
            return {'index': index, 'img': img, 'label': label}
        return {'index': index, 'img': img, 'txt': txt, 'label': label}

    def query(self):
        self.step = 'query'

    def retrieval(self):
        self.step = 'retrieval'

    def get_all_label(self):
        if self.step is 'query':
            return torch.Tensor(self.query_label)
        if self.step is 'retrieval':
            return torch.Tensor(self.retrieval_label)
