import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
# from main import dataset
import AutoEncoder
import dataset
import pandas as pd
from torchvision import transforms
import random

failure_type = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full", "none"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# get rotate matrix
def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]]).to(device)


# rotate the image
def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x


netE = AutoEncoder.get_encoder(1, 32, device)
netD = AutoEncoder.get_decoder(1, 32, device)
restore_ckpt_path = os.path.join('results', str(max(int(step) for step in os.listdir('results'))))
netE.restore(restore_ckpt_path)
netD.restore(restore_ckpt_path)
netE.eval()
netD.eval()

for item in failure_type:
    print(item)
    trainData = pd.read_pickle("train.pkl")
    FailureData = trainData[trainData["failureType"] == item]["waferMap"]
    num = len(FailureData)
    length = len(trainData)
    print(num)

    i = 0
    while i + num <= 5000:
        for imaaage in FailureData:
            resize1 = transforms.Resize([256, 256])
            resize2 = transforms.Resize([64, 64])
            imaaage = transforms.ToPILImage()(imaaage)
            imaaage = resize1(imaaage)
            imaaage = np.array(imaaage)
            imgg = imaaage * 127

            plt.imshow(imgg, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            plt.show()

            imaaage = torch.tensor(imaaage * 127)
            imaaage = imaaage.to(device).float().view((-1, 1, 256, 256))
            latent = netE(imaaage)
            noise = torch.randn((1, 32, 16, 16)).to(device).float()
            latent = latent + 0.5 * noise
            img = netD(latent)
            img = rot_img(img, random.randint(-180, 180), torch.float32)
            img = img.view((256, 256)).to("cpu").float()
            img[img < 110] = 0
            img[img > 200] = 2
            img[img > 110] = 1
            img = img.detach().numpy()
            img = transforms.ToPILImage()(img)
            img = resize2(img)
            img = np.array(img)

            plt.imshow(img*127, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            plt.show()

            trainData.loc[length + i] = [1683, item, "lot1", "Training", 0, img]

            i = i + 1
            if i + num > 5000:
                break
    f = open('train.pkl', 'wb')
    pickle.dump(trainData, f)
    f.close()
