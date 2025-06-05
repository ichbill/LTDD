import os.path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pickle
from PIL import Image
import torchvision
from torchvision import transforms
from PIL import Image


class ImageNetImbanlance(Dataset):
    def __init__(self, imbanlance_rate=0.1, file_path="data/ImageNet/", num_cls=1000, transform=None,
                 train=True):
        # self.root_dir = file_path
        self.classes = sorted(os.listdir(file_path))
        # breakpoint()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.image_paths = [os.path.join(root, filename) for root, _, files in os.walk(file_path) for filename in files if filename.endswith('.JPEG')]
        self.transform = transform

        # self.transform = transform
        assert 0.0 < imbanlance_rate < 1, "imbanlance_rate must 0.0 < p < 1"
        self.num_cls = num_cls
        self.file_path = file_path
        self.imbanlance_rate = imbanlance_rate

        if train is True:
            self.data = self.produce_imbanlance_data(self.imbanlance_rate)
        else:
            self.data = self.produce_test_data()
        self.x_path = self.data['x_path']
        self.y = self.data['y']
        self.targets = self.data['y']

    def __len__(self):
        return len(self.x_path)

    def __getitem__(self, item):
        x_path, y = self.x_path[item], self.y[item]
        x = Image.open(x_path)

        # x = Image.fromarray(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def get_per_class_num(self):
        return self.per_class_num

    def produce_test_data(self):
        with open(os.path.join(self.file_path, "test"), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            x_test = dict[b'data'].reshape([-1, 3, 64, 64]).transpose(0, 2, 3, 1)
            y_test = dict[b'fine_labels']
        dataset = {
            "x": x_test,
            "y": y_test,
        }

        return dataset

    def produce_imbanlance_data(self, imbanlance_rate):
        # train_data = torchvision.datasets.ImageFolder(
        #     os.path.join(self.file_path, 'train')
        # )
        # print('Dataset created')
        x_train = np.array(self.image_paths)
        # breakpoint()
        y_train = [x.split('/')[-2] for x in x_train]
        y_train = np.array([self.class_to_idx[x] for x in y_train])

        data_x = None
        data_y = None

        data_percent = []
        data_num = int(len(x_train) / self.num_cls)

        for cls_idx in range(self.num_cls):
            math_x = (imbanlance_rate ** (cls_idx / (self.num_cls - 1)))
            num = data_num * math_x
            data_percent.append(int(num))

        self.per_class_num = data_percent
        print("imbalance factor is {}".format(data_percent[0] / data_percent[-1]))
        print("per class num：{}".format(data_percent))
        

        for i in range(1, self.num_cls + 1):
            a1 = y_train >= i - 1
            a2 = y_train < i
            index = a1 & a2

            task_train_x = x_train[index]
            label = y_train[index]
            data_num = task_train_x.shape[0]
            if data_percent[i - 1] > data_num:
                index = range(data_num)
            else:
                rng = np.random.RandomState(42)
                index = rng.choice(data_num, data_percent[i - 1], replace=False)
            tem_data = task_train_x[index]
            tem_label = label[index]

            if data_x is None:
                data_x = tem_data
                data_y = tem_label
            else:
                data_x = np.concatenate([data_x, tem_data], axis=0)
                data_y = np.concatenate([data_y, tem_label], axis=0)

        dataset = {
            "x_path": data_x,
            "y": data_y.tolist(),
        }

        return dataset
