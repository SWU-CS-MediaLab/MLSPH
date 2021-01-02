# coding: utf-8
# @Time     : 
# @Author   : Godder
# @Github   : https://github.com/WangGodder

# use np matrix
import numpy as np
# use torch and torch transform utils
import torch
from .dataset import DatasetMirflckr25KTrain, random_index


class Dataset25KPairTrain(DatasetMirflckr25KTrain):
    """
    output a pair of data, the query data is unique, and the another data is random choice from train set.
    you may need to calculate the similar matrix of two output instance.
    """
    def __init__(self, img_dir: str, img_mat_url: str, tag_mat_url: str, label_mat_url: str, single_model=True, batch_size=128, train_num=10000, query_num=2000, random_index=random_index):
        super(Dataset25KPairTrain, self).__init__(img_dir, img_mat_url, tag_mat_url, label_mat_url, single_model, batch_size, train_num, query_num, random_index)
        self.ano_random_item = []
        self.re_random_item()

    def get_random_item(self, item):
        return self.random_item[item // self.batch_size][item % self.batch_size], self.ano_random_item[item // self.batch_size][item % self.batch_size]

    def re_random_item(self):
        self.random_item = []
        self.ano_random_item = []
        for _ in range(self.train_num // self.batch_size):
            random_ind1 = np.random.permutation(range(self.train_num))
            # random_ind2 = np.random.permutation(range(self.train_num))
            self.random_item.append(random_ind1[:self.batch_size])
            self.ano_random_item.append(random_ind1[self.batch_size : self.batch_size * 2])

    def __getitem__(self, item):
        item, ano_item = self.get_random_item(item)
        # ano_item = np.random.choice(np.setdiff1d(range(self.train_num), item), 1)[0]
        img = txt = None
        if self.img_read:
            img = self.read_img(item)
            ano_img = self.read_img(ano_item)
        if self.txt_read:
            txt = torch.Tensor(self.txt[item][np.newaxis, :, np.newaxis])
            ano_txt = torch.Tensor(self.txt[ano_item][np.newaxis, :, np.newaxis])
        label = torch.Tensor(self.label[item])
        ano_label = torch.Tensor(self.label[ano_item])
        index = torch.from_numpy(np.array(item))
        ano_index = torch.from_numpy(np.array(ano_item))
        if self.img_read is False:
            return {'index': index, 'ano_index': ano_index, 'txt': txt, 'ano_txt': ano_txt, 'label': label, 'ano_label': ano_label}
        if self.txt_read is False:
            return {'index': index, 'ano_index': ano_index, 'img': img, 'ano_img': ano_img, 'label': label, 'ano_label': ano_label}


def get_pair_train_set(img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=128, train_num=10000, query_num=2000):
    return Dataset25KPairTrain(img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=batch_size, train_num=train_num, query_num=query_num)
