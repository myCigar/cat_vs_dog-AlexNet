import os
import torch.utils.data as data
from scipy.misc import imread
from PIL import Image


class CatDogDataset(data.Dataset):
    def __init__(self, args, mode='train', transform=None):
        self.args = args
        self.transform = transform
        self.mode = mode
        self.names = self.__dataset_info()

    def __getitem__(self, index):
        x = imread(self.args.data_path + "/" + self.names[index], mode='RGB') # numpy
        x = Image.fromarray(x) # PIL

        x_label = 0 if 'cat' in self.names[index] else 1

        if self.transform is not None:
            x = self.transform(x)

        return x, x_label

    def __len__(self):
        return len(self.names)

    # 取train中前500张的猫和狗图片为测试集，所以一共有1000张测试集，24000张训练集
    def __dataset_info(self):
        img_path = self.args.data_path
        imgs = [f for f in os.listdir(img_path) if
                os.path.isfile(os.path.join(img_path, f)) and f.endswith('.jpg')]

        names = []
        for name in imgs:
            index = int(name.split('.')[1])
            # train dataset
            if self.mode == 'train':
                if index >= 500:
                    names.append(name)
            # test dataset: 1000 imgs
            elif self.mode == 'test':
                if index < 500:
                    names.append(name)

        return names


class TestDataset(data.Dataset):
    def __init__(self, args, transform=None):
        self.args = args
        self.transform = transform
        self.names = self.__dataset_info()

    def __getitem__(self, index):
        x = imread(self.args.data_path + "/" + str(index+1) + '.jpg', mode='RGB')  # numpy
        x = Image.fromarray(x)  # PIL

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.names)

    # 取train中前500张的猫和狗图片为测试集，所以一共有1000张测试集，24000张训练集
    def __dataset_info(self):
        img_path = self.args.data_path
        imgs = [f for f in os.listdir(img_path) if
                os.path.isfile(os.path.join(img_path, f)) and f.endswith('.jpg')]

        names = []
        for name in imgs:
            names.append(name)

        return names


