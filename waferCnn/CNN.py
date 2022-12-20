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


def get_classifier(num_channels, num_classes, device):
    model = Classifier(num_channels, num_classes).to(device)
    model.apply(weights_init)
    return model


class Classifier(nn.Module):
    def __init__(self, num_channels=1, num_classes=9):
        super(Classifier, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        # B, 1, 256, 256
        self.conv4 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(num_features=32)
        # self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.lrelu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        # B , 32, 128 , 128
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn5 = nn.BatchNorm2d(num_features=64)
        # self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.lrelu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        # 64 64 64
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn6 = nn.BatchNorm2d(num_features=128)
        # self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.lrelu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size=2)
        # 128 32 32
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=128)
        # self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.lrelu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 128 16 16
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=512)
        # self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.lrelu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        #512 8 8
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=2048)
        # self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        self.lrelu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # 2048 4 4
        self.FC1 = nn.Linear(32*32*32, 1*256)
        self.relu1 = nn.ReLU()
        self.FC2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.FC3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.pre = nn.Linear(64, 1*num_classes)
        self.sel = nn.Linear(64, 1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.float()

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.lrelu5(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.lrelu6(x)
        x = self.pool6(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu3(x)
        x = self.pool3(x)

        x = self.FC1(x.view((-1, 32*32*32)))
        x = self.relu1(x)
        x = self.FC2(x)
        x = self.relu2(x)
        x = self.FC3(x)
        x = self.relu3(x)
        pre = self.pre(x)
        # pre = self.relu1(pre)
        # pre = self.softmax(pre)
        return pre

    def restore(self, ckpt_dir):
        try:
            if os.path.exists(os.path.join(ckpt_dir, 'classifier.bin')):
                path = os.path.join(ckpt_dir, 'classifier.bin')
            else:
                path = os.path.join(ckpt_dir, str(max(int(name) for name in os.listdir(ckpt_dir))), 'classifier.bin')
        except:
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, ckpt_dir, global_step):
        os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
        path = os.path.join(ckpt_dir, str(global_step), 'classifier.bin')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]
