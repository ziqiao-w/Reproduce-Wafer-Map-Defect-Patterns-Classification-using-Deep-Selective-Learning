import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import pandas as pd
import pickle


class CNN_set(Dataset):
    def __init__(self, img_path, batch_size):
        self.image_files = pd.read_pickle(img_path)
        self.batch_size = batch_size
        self.trainImg = self.image_files["waferMap"]
        self.trainIdx = self.trainImg.index
        print("num of training img: " + str(len(self.trainImg)))
        self.trainLabel = self.image_files["failureType"]
        print("num of training label: " + str(len(self.trainLabel)))

    def __len__(self):
        return len(self.trainImg)

    def __getitem__(self, index):
        idx = self.trainIdx[index]
        img = self.trainImg[idx]
        resize = transforms.Resize([256, 256])
        img = transforms.ToPILImage()(img)
        img = resize(img)
        img = np.array(img)
        img = img * 127

        onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        label = self.trainLabel[idx]
        label_dict = {"Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge-Ring": 3, "Loc": 4,
                      "Random": 5, "Scratch": 6, "Near-full": 7, "none": 8}
        clas = label_dict[label]
        onehot[clas] = 1
        onehot = np.array(onehot)
        return img, onehot

    @property
    def training_loader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=True)


class CNN_test(Dataset):

    def __init__(self, img_path):
        self.image_files = pd.read_pickle(img_path)
        self.testIdx = self.image_files[self.image_files['trainTestLabel'] == 'Test'].index
        print("num of test img: " + str(len(self.testIdx)))
        # self.testFailureType = self.image_files.loc[self.testIdx, 'failureType']

    def __len__(self):
        return len(self.testIdx)

    def __getitem__(self, index):
        idx = self.testIdx[index]
        img = self.image_files["waferMap"][idx]
        resize = transforms.Resize([256, 256])
        img = transforms.ToPILImage()(img)
        img = resize(img)
        img = np.array(img)
        img = img * 127

        onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        label = self.image_files['failureType'][idx]
        label_dict = {"Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge-Ring": 3, "Loc": 4,
                      "Random": 5, "Scratch": 6, "Near-full": 7, "none": 8}
        clas = label_dict[label]
        onehot[clas] = 1
        onehot = np.array(onehot)
        return img, onehot

    @property
    def test_loader(self):
        return DataLoader(self, batch_size=64, shuffle=True)
