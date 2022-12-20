import AutoEncoder
from trainer import Trainer
# from dataset import Dataset
import dataset
from tensorboardX import SummaryWriter

# from pytorch_fid import fid_score

import torch
import torch.optim as optim
import os
import argparse

if __name__ == "__main__":
    tb_writer = SummaryWriter(log_dir='./runs')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = dataset.A_Dataset("WM811K.pkl")
    netE = AutoEncoder.get_encoder(num_channels=1, latent_dim=32, device=device)
    netD = AutoEncoder.get_decoder(num_channels=1, latent_dim=32, device=device)
    optimE = optim.Adam(netE.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    trainer = Trainer(device, netE, netD, optimE, optimD, dataset, ckpt_dir='results', tb_writer=tb_writer)
    trainer.train(epochs=20, logging_steps=10, saving_steps=1000)
