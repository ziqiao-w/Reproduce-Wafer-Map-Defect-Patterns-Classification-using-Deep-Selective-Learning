import CNN
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
    dataset = dataset.CNN_set("train.pkl", batch_size=32)
    netC = CNN.get_classifier(num_channels=1, num_classes=9, device=device)
    optimC = optim.Adam(netC.parameters(), lr=0.0002, betas=(0.5, 0.999))
    trainer = Trainer(device, netC, optimC, dataset, ckpt_dir='results', tb_writer=tb_writer)
    trainer.train(epochs=20, logging_steps=50, saving_steps=1000)
