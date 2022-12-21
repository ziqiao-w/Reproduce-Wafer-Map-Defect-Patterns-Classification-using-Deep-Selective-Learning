import torch
import torch.nn as nn
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from cnn import Model

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=100,
    help='Batch size for mini-batch training and evaluating. Default: 100')
parser.add_argument('--num_epochs', type=int, default=20,
    help='Number of training epoch. Default: 20')
parser.add_argument('--learning_rate', type=float, default=1e-3,
    help='Learning rate during optimization. Default: 1e-3')
parser.add_argument('--coverage', type=float, default=0.2,
    help='Coverage for Selectiveloss. Default: 2e-1')
parser.add_argument('--lamda', type=float, default=0.5,
    help='Lamda for Selectiveloss. Default: 0.5')
parser.add_argument('--alpha', type=float, default=0.5,
    help='alpha for Loss funct in selective net. Default: 0.5')
parser.add_argument('--is_train', type=bool, default=False,
    help='True to train and False to inference. Default: True')
parser.add_argument('--data_dir', type=str, default='./LSWMD.pkl',
    help='Data directory. Default: ./LSWMD.pkl')
parser.add_argument('--train_dir', type=str, default='./train',
    help='Training directory for saving model. Default: ./train')
parser.add_argument('--inference_version', type=int, default=0,
	help='The version for inference. Set 0 to use latest checkpoint. Default: 0')

err_type = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']
cnt = [0,0,0,0,0,0,0,0,0]
cnt_ = [0,0,0,0,0,0,0,0,0]
args = parser.parse_args()



def shuffle(X, y, shuffle_parts):
    chunk_size = int(len(X) / shuffle_parts)
    shuffled_range = list(range(chunk_size))

    X_buffer = np.copy(X[0:chunk_size])
    y_buffer = np.copy(y[0:chunk_size])

    for k in range(shuffle_parts):
        np.random.shuffle(shuffled_range)
        for i in range(chunk_size):
            X_buffer[i] = X[k * chunk_size + shuffled_range[i]]
            y_buffer[i] = y[k * chunk_size + shuffled_range[i]]

        X[k * chunk_size:(k + 1) * chunk_size] = X_buffer
        y[k * chunk_size:(k + 1) * chunk_size] = y_buffer

    return X, y


def train_epoch(model, X, y, optimizer): # Training Process
    model.train()
    loss, acc, sel_acc, sel_rate = 0.0, 0.0, 0.0, 0.0
    st, ed, times = 0, args.batch_size, 0
    while st < len(X) and ed <= len(X):
        optimizer.zero_grad()
        X_batch, y_batch = torch.from_numpy(X[st:ed]).to(device), torch.from_numpy(y[st:ed]).to(device)
        loss_, acc_, sel_rate_, sel_acc_ = model(X_batch, y_batch)
        loss_.backward()
        optimizer.step()

        loss += loss_.cpu().data.numpy()
        acc += acc_.cpu().data.numpy()
        sel_rate += sel_rate_
        sel_acc += sel_acc_
        st, ed = ed, ed + args.batch_size
        times += 1
    loss /= times
    acc /= times
    sel_rate /= times
    sel_acc /= times
    return acc, loss, sel_rate, sel_acc

def valid_epoch(model, X, y): # Valid Process
    model.eval()
    loss, acc, sel_acc, sel_rate = 0.0, 0.0, 0.0, 0.0
    st, ed, times = 0, args.batch_size, 0
    while st < len(X) and ed <= len(X):
        X_batch, y_batch = torch.from_numpy(X[st:ed]).to(device), torch.from_numpy(y[st:ed]).to(device)
        loss_, acc_, sel_rate_, sel_acc_ = model(X_batch, y_batch)

        loss += loss_.cpu().data.numpy()
        acc += acc_.cpu().data.numpy()
        sel_rate += sel_rate_
        sel_acc += sel_acc_
        st, ed = ed, ed + args.batch_size
        times += 1
    loss /= times
    acc /= times
    sel_rate /= times
    sel_acc /= times
    return acc, loss, sel_rate, sel_acc

def trans_label(x):
    y = type(x)
    if (y == str):
        return x
    return x[0]

def trans_failt(x):
    y = type(x)
    if(y == str):
        return x
    return x[0][0]

def inference(model, X): # Test Process
    model.eval()
    pred_ = model(torch.from_numpy(X).to(device))
    return pred_

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)

    X_train = []
    X_test  = []
    Y_train = []
    Y_test  = []
    data = pd.read_pickle(args.data_dir)
    trainLabel = data.get('trainTestLabel')
    if type(trainLabel) == type(None):
        trainLabel = data['trianTestLabel']
    i = 0
    for _ in range(0,len(data)):
        while type(data['failureType'].get(i)) == type(None) :
            i = i + 1
        # print(i,data['failureType'][i])
        # print(type(data['failureType'][i]))
        
        if type(data['failureType'][i]) != str: 
            if data['failureType'][i].size == 0:
                i = i + 1
                continue
        if type(trainLabel[i]) != str and trainLabel[i].size == 0:
            i = i + 1
            continue
        # print(trans_label(trainLabel[i]),i)
        if trans_label(trainLabel[i]) == 'Training' :
            img_tmp = torch.tensor(data['waferMap'][i])
            img_tmp = img_tmp.reshape(1,1,img_tmp.shape[0],-1)
            img_tmp = torch.nn.functional.interpolate(img_tmp, (256,256))
            img_tmp = 127*img_tmp.reshape(1,256,256).float()
            X_train.append(np.array(img_tmp))
            label_tmp = torch.zeros(9)
            for j in range(0,9):
                if err_type[j] == trans_failt(data['failureType'][i]):
                    label_tmp[j] = 1.0
                    cnt[j]=cnt[j]+1
            Y_train.append(np.array(label_tmp))
        if trans_label(trainLabel[i]) == 'Test' :
            img_tmp = torch.tensor(data['waferMap'][i])
            img_tmp = img_tmp.reshape(1,1,img_tmp.shape[0],-1)
            img_tmp = torch.nn.functional.interpolate(img_tmp, (256,256))
            img_tmp = 127*img_tmp.reshape(1,256,256).float()
            X_test.append(np.array(img_tmp))

            label_tmp = torch.zeros(9)
            for j in range(0,9):
                if err_type[j] == trans_failt(data['failureType'][i]):
                    label_tmp[j] = 1.0
                    cnt_[j] = cnt_[j] + 1
            Y_test.append(np.array(label_tmp))
        i = i + 1
    print(cnt)
    print(cnt_)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    X_train, Y_train = shuffle(X_train, Y_train, 1)
    X_test, Y_test = shuffle(X_test, Y_test, 1)
    sz = int(0.8*len(X_train))
    X_val, Y_val = X_train[sz:],Y_train[sz:]
    X_train, Y_train = X_train[:sz],Y_train[:sz]
    if args.is_train:
        
        cnn_model = Model()
        cnn_model.to(device)
        print(cnn_model)
        optimizer = optim.Adam(cnn_model.parameters(), lr = args.learning_rate)

        pre_losses = [1e18] * 3
        best_val_acc = 0.0

        for epoch in range(1, args.num_epochs+1):
            start_time = time.time()
            train_acc, train_loss, _, _ = train_epoch(cnn_model, X_train, Y_train, optimizer)
            X_train, Y_train = shuffle(X_train, Y_train, 1)

            val_acc, val_loss, val_sel, val_sel_acc = valid_epoch(cnn_model, X_val, Y_val)
            
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                with open(os.path.join(args.train_dir, 'checkpoint_{}.pth.tar'.format(epoch)), 'wb') as fout:
                    torch.save(cnn_model, fout)
                with open(os.path.join(args.train_dir, 'checkpoint_0.pth.tar'), 'wb') as fout:
                    torch.save(cnn_model, fout)

            epoch_time = time.time() - start_time
            print("Epoch " + str(epoch) + " of " + str(args.num_epochs) + " took " + str(epoch_time) + "s")
            print("  learning rate:                 " + str(optimizer.param_groups[0]['lr']))
            print("  training loss:                 " + str(train_loss))
            print("  training accuracy:             " + str(train_acc))
            print("  validation loss:               " + str(val_loss))
            print("  validation accuracy:           " + str(val_acc))
            print("  validation select:             " + str(val_sel))
            print("  validation select accuracy:    " + str(val_sel_acc))
            print("  best epoch:                    " + str(best_epoch))
            print("  best validation accuracy:      " + str(best_val_acc))
            if train_loss > max(pre_losses):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9995
            pre_losses = pre_losses[1:] + [train_loss]
    else :
        print("begin testing")
        cnn_model = Model()
        cnn_model.to(device)
        model_path = os.path.join(args.train_dir, 'checkpoint_%d.pth.tar' % args.inference_version)
        if os.path.exists(model_path):
            cnn_model = torch.load(model_path)
        print(args.data_dir)

        count = 0
        Y_test = torch.argmax(torch.from_numpy(Y_test),1).int()
        for i in range(len(X_test)):
            test_image = X_test[i].reshape((1, 1, 256, 256))
            result = inference(cnn_model, test_image)[0]
            result = torch.argmax(result,1).int()
            # stss = str(result) + " " +str(Y_test[i])
            # print(stss)
            if result == Y_test[i]:
                count += 1
        print("test accuracy: {}".format(float(count) / len(X_test)))
