import torch.nn as nn
import torch
import os


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def get_encoder(num_channels, latent_dim, device):
    model = Encoder(num_channels, latent_dim).to(device)
    model.apply(weights_init)
    return model


def get_decoder(num_channels, latent_dim, device):
    model = Decoder(num_channels, latent_dim).to(device)
    model.apply(weights_init)
    return model


class Encoder(nn.Module):
    def __init__(self, num_channels=1, latent_dim=32):
        super(Encoder, self).__init__()
        self.num_channels = num_channels
        self.latent_dim = latent_dim

        # TODO START
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=5, stride=2, padding=(2, 1)),
        #     nn.BatchNorm2d(num_features=64),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=(2, 1)),
        #     nn.BatchNorm2d(num_features=32),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=(2, 1)),
        #     nn.BatchNorm2d(num_features=32),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=latent_dim, kernel_size=5, stride=2, padding=(2, 1)),
        #     nn.BatchNorm2d(num_features=latent_dim),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.ReLU()
        # )
        # TODO END

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.poo1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.poo2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.poo3 = nn.MaxPool2d(kernel_size=2)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=latent_dim, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(num_features=latent_dim)
        self.poo4 = nn.MaxPool2d(kernel_size=2)
        self.relu4 = nn.ReLU()
    def forward(self, x):
        '''
        *   Arguments:
            *   z (torch.FloatTensor): [batch_size, latent_dim, 1, 1]
        '''
        x = x.float()
        x = x.to(next(self.parameters()).device)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.poo1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.poo2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.poo3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.poo4(x)
        x = self.relu4(x)
        # x = self.encoder(x)
        return x

    def restore(self, ckpt_dir):
        try:
            if os.path.exists(os.path.join(ckpt_dir, 'encoder.bin')):
                path = os.path.join(ckpt_dir, 'encoder.bin')
            else:
                path = os.path.join(ckpt_dir, str(max(int(name) for name in os.listdir(ckpt_dir))), 'encoder.bin')
        except:
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, ckpt_dir, global_step):
        os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
        path = os.path.join(ckpt_dir, str(global_step), 'encoder.bin')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]


class Decoder(nn.Module):
    def __init__(self, num_channels=1, latent_dim=32):
        super(Decoder, self).__init__()
        self.num_channels = num_channels
        self.latent = latent_dim

        # self.decoder = nn.Sequential(
        #     # input is (num_channels) x 32 x 32
        #     nn.Conv2d(num_channels + 1, hidden_dim, kernel_size=(5, 4), stride=(1, 2), padding=(0, 1), bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(hidden_dim, 2 * hidden_dim, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (hidden_dim) x 16 x 16
        #     nn.Conv2d(2 * hidden_dim, hidden_dim * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(hidden_dim * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (hidden_dim*2) x 8 x 8
        #     nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(hidden_dim * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (hidden_dim*4) x 4 x 4
        #     nn.Conv2d(hidden_dim * 8, 1, 4, 1, 0, bias=False),
        #     nn.Sigmoid()
        # )
        # self.decoder = nn.Sequential(
        #
        #         nn.ConvTranspose2d(in_channels=latent_dim, out_channels=32, kernel_size=4,
        #                            stride=2, padding=1),
        #         nn.BatchNorm2d(num_features=32),
        #         nn.ReLU(),
        #         nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4,
        #                            stride=2, padding=1),
        #         nn.BatchNorm2d(num_features=32),
        #         nn.ReLU(),
        #         nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
        #                            padding=1),
        #         nn.BatchNorm2d(num_features=64),
        #         nn.ReLU(),
        #         nn.ConvTranspose2d(in_channels=64, out_channels=num_channels, kernel_size=4,
        #                            stride=2,
        #                            padding=1),
        #         nn.BatchNorm2d(num_features=num_channels),
        #         nn.ReLU()
        #         # nn.Tanh()
        # )

        self.conv1 = nn.ConvTranspose2d(in_channels=latent_dim, out_channels=32, kernel_size=4,
                               stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4,
                               stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                               padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=num_channels, kernel_size=4,
                               stride=2,
                               padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=num_channels)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = x.float()
        x = x.to(next(self.parameters()).device)
        # x = self.decoder(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        return x

    def restore(self, ckpt_dir):
        try:
            if os.path.exists(os.path.join(ckpt_dir, 'decoder.bin')):
                path = os.path.join(ckpt_dir, 'decoder.bin')
            else:
                path = os.path.join(ckpt_dir, str(max(int(name) for name in os.listdir(ckpt_dir))), 'decoder.bin')
        except:
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, ckpt_dir, global_step):
        os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
        path = os.path.join(ckpt_dir, str(global_step), 'decoder.bin')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]
