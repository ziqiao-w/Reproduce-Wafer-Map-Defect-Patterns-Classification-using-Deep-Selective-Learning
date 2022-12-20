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
import CNN
import dataset
import pandas as pd
from torchvision import transforms
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

netC = CNN.get_classifier(1, 9, device)
best_step = 0
best_acc = 0
# restore_ckpt_path = os.path.join('results', str(max(int(step) for step in os.listdir('results'))))
for step in os.listdir('results'):
    print("STEP", int(step))



    restore_ckpt_path = os.path.join('results', str(int(step)))
    netC.restore(restore_ckpt_path)

    netC.eval()

    testset = dataset.CNN_test("WM811K.pkl")

    dataLoader = testset.test_loader
    i = 0
    acc_total = 0
    for data in dataLoader:
        test_imgs = data[0].to(device)
        test_imgs = test_imgs.view((-1, 1, 256, 256))
        test_imgs = test_imgs.float()
        test_labels = data[1].to(device)
        test_labels = test_labels.view((-1, 9)).float()
        onehot = netC(test_imgs)
        prediction = torch.argmax(onehot, 1)
        truth = torch.argmax(test_labels, 1)
        correct = (prediction == truth).sum().float()
        total = len(truth)
        acc = ((correct / total).cpu().detach().data.numpy())
        acc_str = 'Accuracy: %f' % acc
        acc_total += acc
        print(acc_str)
        i += 1

    print("Total Accuracy : %f" % (acc_total/i))
    if (acc_total/i) > best_acc:
        best_step = int(step)
        best_acc = (acc_total/i)
print("BEST Accuracy : %f" % best_acc)
print("BEST STEP : %f" % best_step)
