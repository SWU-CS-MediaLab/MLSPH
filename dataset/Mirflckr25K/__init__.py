# -*- coding: utf-8 -*-
# @Time    : 2019/5/17
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from .dataset import DatasetMirflckr25KTrain, DatasetMirflckr25KValid
from .pair_dataset import get_pair_train_set
from .triplet_dataset import get_triple_train_set


__all__ = ['get_single_train_set', 'get_pair_train_set', 'get_valid_set', 'get_triple_train_set']


def get_single_train_set(img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=128, train_num=10000, query_num=2000):
    return DatasetMirflckr25KTrain(img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=batch_size, train_num=train_num, query_num=query_num)


def get_valid_set(img_dir, img_mat_url, tag_mat_url, label_mat_url, query_num=2000):
    return DatasetMirflckr25KValid(img_dir, img_mat_url, tag_mat_url, label_mat_url, query_num=query_num)