import torch 
from torch import nn
from torch.nn.parameter import Parameter
import numpy as np

class select_loss(nn.Module):
    def __init__(self, cov=0.8, lamda = 0.1):
        super().__init__()
        self.cov = cov
        self.lamda = lamda
        self.loss = nn.CrossEntropyLoss()

    def forward(self, predict, select, label):
        em_cov = select.mean()
        em_risk = (self.loss(predict, label) * select.view(-1)).mean()

        penalty = np.maximum(0.0, self.cov - float(em_cov)) ** 2
        return em_risk + self.lamda * penalty


class Model(nn.Module):
    def __init__(self, cov=0.8, lamda = 0.1, alpha = 0.9):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding = 2, bias = False),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 32, 3, padding = 1, bias = False),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 32, 3, padding = 1, bias = False),
            nn.MaxPool2d(2,2)
        )
        self.fc1 = nn.Linear(32768, 256)
        self.predict = nn.Sequential(
            nn.Linear(256,9)
        )
        
        self.select = torch.nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.loss1 = nn.CrossEntropyLoss()
        self.loss = select_loss(cov, lamda)
        self.alpha = alpha
    
    def forward(self, x, y = None):
        x_cov = self.cnn(x)
        x_out = self.fc1(x_cov.view(x_cov.size(0),-1))
        
        x_predict = self.predict(x_out)
        x_select = self.select(x_out)
        # print(x_predict)
        if y is None :
            return x_predict, x_select
        y = torch.argmax(y, 1)
        # loss = self.alpha * self.loss(x_predict, x_select, y) + (1 - self.alpha) * self.loss1(x_predict, y)
        loss = self.loss1(x_predict,y)
        
        pred = torch.argmax(x_predict, 1)
        # print(pred, x_select, y)
        # print(y,pred)
        acc = torch.mean((pred.int() == y.int()).float())
        sel_rate = 0
        sel_acc = 0

        x_select=(x_select+0.5).int()
        for i in range(len(x_select)):
            if x_select[i] == 1:
                sel_rate += 1
                if pred[i].int() == y[i].int():
                    sel_acc += 1
        if sel_rate == 0 :
            sel_acc = 0
        else :
            sel_acc = sel_acc/ sel_rate
        sel_rate /= len(x_select)
        return loss, acc, sel_rate, sel_acc
