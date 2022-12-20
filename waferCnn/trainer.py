import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
from dataset import CNN_set

class Trainer(object):
    def __init__(self, device, netC, optimC, dataset, ckpt_dir, tb_writer):
        self._device = device
        self._netC = netC
        self._optimC = optimC
        self._dataset = dataset
        self._ckpt_dir = ckpt_dir
        self._tb_writer = tb_writer
        os.makedirs(ckpt_dir, exist_ok=True)
        self._netC.restore(ckpt_dir)

    def train_step(self, real_img, label):

        onehot = self._netC(real_img)
        prediction = torch.argmax(onehot, 1)
        truth = torch.argmax(label, 1)
        correct = (prediction == truth).sum().float()
        total = len(truth)
        acc = ((correct / total).cpu().detach().data.numpy())
        # acc_str = 'Accuracy: %f' % ((correct / total).cpu().detach().data.numpy())
        self._netC.zero_grad()

        CEloss = nn.CrossEntropyLoss()

        Loss = CEloss(onehot, truth)
        Loss.backward()

        # grad = list(self._netC.children())[0].weight.grad

        self._optimC.step()
        # return what are specified in the docstring
        return Loss, acc

    def train(self, epochs, logging_steps, saving_steps):

        # iterator = iter(cycle(self._dataset.training_loader))
        dataloader = self._dataset.training_loader
        # for i in tqdm(range(num_training_updates), desc='Training'):
        i = 0
        train_Loss = 0
        train_Acc = 0
        for epoch in range(0, epochs):
            for data in dataloader:

                # inp, _ = next(iterator)
                self._netC.train()  # 开启train模式，让batchnorm层可以工作
                real_imgs = data[0].to(self._device)
                real_imgs = real_imgs.view((-1, 1, 256, 256))
                real_imgs = real_imgs.float()

                real_labels = data[1].to(self._device)
                real_labels = real_labels.view((-1, 9)).float()

                loss, acc = self.train_step(real_img=real_imgs, label=real_labels)
                train_Loss += loss.item()
                train_Acc += acc
                if (i + 1) % logging_steps == 0:
                    train_Loss = train_Loss / logging_steps
                    train_Acc = train_Acc / logging_steps
                    acc_str = 'Accuracy: %f' % train_Acc
                    self._tb_writer.add_scalar("CEloss", train_Loss, global_step=i)
                    print("Loss", train_Loss)
                    print(acc_str)
                    train_Loss = 0
                    train_Acc = 0

                if (i + 1) % 1000 == 0:
                    dirname = self._netC.save(self._ckpt_dir, i)
                    # self._netC.eval()
                    # noise = self._netE(real_imgs[0].view(1, 1, 256, 256))
                    # imgs = self._netD(noise)
                    #
                    # plt.imsave("{}_{}.png".format(epoch, i), imgs.squeeze().to("cpu").detach().numpy(),
                    #                cmap='gray', vmin=0, vmax=255)
                    # plt.imshow(imgs.squeeze().to("cpu").detach().numpy(), cmap='gray', vmin=0, vmax=255)
                    # plt.axis('off')
                    # plt.show()
                    # testLoader = A_Test("WM811K.pkl").test_loader
                    # for tst in testLoader:
                    #     tstimg = tst.to(self._device)
                    #     tstimg = (tstimg.view((-1, 1, 256, 256))).float()
                    #     fkimg = self._netE(tstimg)
                    #     fkimg = self._netD(fkimg)
                    #     MSEloss = nn.MSELoss()
                    #     test_loss = MSEloss(tstimg, fkimg)
                    #     print("Test Loss")
                    #     print(test_loss)
                    #     break
                i = i + 1
