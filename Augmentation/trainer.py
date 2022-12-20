import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
from dataset import A_Test

class Trainer(object):
    def __init__(self, device, netE, netD, optimE, optimD, dataset, ckpt_dir, tb_writer):
        self._device = device
        self._netE = netE
        self._netD = netD
        self._optimE = optimE
        self._optimD = optimD
        self._dataset = dataset
        self._ckpt_dir = ckpt_dir
        self._tb_writer = tb_writer
        os.makedirs(ckpt_dir, exist_ok=True)
        self._netE.restore(ckpt_dir)
        self._netD.restore(ckpt_dir)

    def train_step(self, real_img):

        latent = self._netE(real_img)
        fake_img = self._netD(latent)
        self._netD.zero_grad()
        self._netE.zero_grad()

        l2loss = nn.MSELoss()
        Loss = l2loss(fake_img, real_img)
        Loss.backward()
        self._optimD.step()
        self._optimE.step()
        # return what are specified in the docstring
        return Loss

    def train(self, epochs, logging_steps, saving_steps):

        # iterator = iter(cycle(self._dataset.training_loader))
        dataloader = self._dataset.training_loader
        # for i in tqdm(range(num_training_updates), desc='Training'):
        i = 0
        for epoch in range(0, epochs):
            for data in dataloader:

                # inp, _ = next(iterator)
                self._netD.train()  # 开启train模式，让batchnorm层可以工作
                self._netE.train()
                real_imgs = data.to(self._device)
                real_imgs = real_imgs.view((-1, 1, 256, 256))
                real_imgs = real_imgs.float()
                loss = self.train_step(real_img=real_imgs)

                if (i + 1) % logging_steps == 0:
                    self._tb_writer.add_scalar("L2loss", loss, global_step=i)
                    print("Loss", loss)

                if (i + 1) % 250 == 0:
                    dirname = self._netD.save(self._ckpt_dir, i)
                    dirname = self._netE.save(self._ckpt_dir, i)
                    self._netE.eval()
                    self._netD.eval()
                    noise = self._netE(real_imgs[0].view(1, 1, 256, 256))
                    imgs = self._netD(noise)

                    plt.imsave("{}_{}.png".format(epoch, i), imgs.squeeze().to("cpu").detach().numpy(),
                                   cmap='gray', vmin=0, vmax=255)
                    plt.imshow(imgs.squeeze().to("cpu").detach().numpy(), cmap='gray', vmin=0, vmax=255)
                    plt.axis('off')
                    plt.show()
                    testLoader = A_Test("WM811K.pkl").test_loader
                    for tst in testLoader:
                        tstimg = tst.to(self._device)
                        tstimg = (tstimg.view((-1, 1, 256, 256))).float()
                        fkimg = self._netE(tstimg)
                        fkimg = self._netD(fkimg)
                        MSEloss = nn.MSELoss()
                        test_loss = MSEloss(tstimg, fkimg)
                        print("Test Loss")
                        print(test_loss)
                        break
                i = i + 1
