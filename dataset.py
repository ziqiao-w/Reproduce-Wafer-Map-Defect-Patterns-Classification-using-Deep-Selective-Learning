import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import pandas as pd
import pickle

class A_Dataset(Dataset):
    def __init__(self, img_path):
        self.image_files = pd.read_pickle(img_path)
        self.trainIdx = self.image_files[self.image_files['trainTestLabel'] == 'Training'].index
        self.trainImg = self.image_files[self.image_files['trainTestLabel'] == 'Training']
        with open("train.pkl", "wb") as f:
            pickle.dump(self.trainImg, f)
            f.close()
        self.testIdx = self.image_files[self.image_files['trainTestLabel'] == 'Test'].index
        self.trainFailureType = self.image_files.loc[self.trainIdx, 'failureType']
        self.testFailureType = self.image_files.loc[self.testIdx, 'failureType']

    def __len__(self):
        return len(self.trainIdx)

    def __getitem__(self, index):
        idx = self.trainIdx[index]
        img = self.image_files["waferMap"][idx]
        resize = transforms.Resize([256, 256])
        img = transforms.ToPILImage()(img)
        img = resize(img)
        img = np.array(img)
        img = img * 127
        return img

    @property
    def training_loader(self):
        return DataLoader(self, batch_size=32, shuffle=True)


class A_Test(Dataset):

    def __init__(self, img_path):
        self.image_files = pd.read_pickle(img_path)
        self.testIdx = self.image_files[self.image_files['trainTestLabel'] == 'Test'].index
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
        return img

    @property
    def test_loader(self):
        return DataLoader(self, batch_size=32, shuffle=True)

