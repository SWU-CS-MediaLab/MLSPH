# coding: utf-8
# @Time     : 
# @Author   : Godder
# @Github   : https://github.com/WangGodder
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                 # transforms.RandomChoice([
                                 #     transforms.ColorJitter(brightness=0.5),
                                 #     transforms.ColorJitter(contrast=0.5),
                                 #     transforms.ColorJitter(hue=0.1),
                                 #     transforms.RandomRotation([0, 0]),
                                 #     transforms.RandomHorizontalFlip(1),
                                 #
                                 # ]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

valid_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])


class CrossModalDatasetBase(Dataset):
    def __init__(self, img_dir: str, img_mat_url: str, tag_mat_url: str, label_mat_url: str, transform, name):
        super(CrossModalDatasetBase, self).__init__()
        if not os.path.exists(img_dir):
            raise FileExistsError(img_dir + " is not exist")
        self.img_dir = img_dir
        if not os.path.exists(img_mat_url):
            raise FileExistsError(img_mat_url + " is not exist")
        if not os.path.exists(tag_mat_url):
            raise FileExistsError(tag_mat_url + " is not exist")
        if not os.path.exists(label_mat_url):
            raise FileExistsError(label_mat_url + " is not exist")
        if not os.path.isdir(img_dir):
            raise NotADirectoryError(img_dir + " is not a dir")

        # transform for images
        self.transform = transform
        print("data set name " + name)
        print(transform)

        self.img_names = None
        self.txt = None
        self.label = None

        self.random_item = []

        self.img_read = True
        self.txt_read = False

    def read_img(self, item):
        image_url = os.path.join(self.img_dir, self.img_names[item])
        image = Image.open(image_url).convert('RGB')
        image = self.transform(image)
        return image

    def img_load(self):
        if self.img_read is False:
            self.img_read = True
            self.txt_read = False

    def txt_load(self):
        if self.txt_read is False:
            self.img_read = False
            self.txt_read = True

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


__all__ = ['train_transform', 'valid_transform', 'CrossModalDatasetBase']


if __name__ == '__main__':
    print(train_transform)