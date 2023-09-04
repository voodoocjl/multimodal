import random
import pickle
import csv
import numpy as np
import torch
from torch import nn, flip
from torch.utils.data import DataLoader, TensorDataset
import time
from sklearn.metrics import accuracy_score, f1_score
from Network import Linear, Mlp, Conv_Net, Attention, get_label,change_code, transform_2d, transform_attention, positional_encoding

# torch.cuda.is_available = lambda : False

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

with open('data/mosi_test', 'rb') as file:
    dataset = pickle.load(file)

arch_code, energy = [], []
for key in dataset:
    arch_code.append(eval(key))
    energy.append(dataset[key])
arch_code = torch.from_numpy(np.asarray(arch_code, dtype=np.float32))
energy =  torch.from_numpy(np.asarray(energy, dtype=np.float32))

# # attention-35
# from Network_old import Attention, get_label,change_code, transform_attention
# arch_code_t = transform_attention(arch_code, [1, 5])   # 5 layers
# model = Attention(3, 1, 1)

# Attention - 5
arch_code_t = transform_attention(arch_code, [5, 1])   # 5 layers
model = Attention(10, 1, 1)

# # linear
# arch_code = change_code(arch_code)
# model = Linear(21, 1)

# # mlp
# arch_code = change_code(arch_code)
# model = Mlp(21, 32, 1)

# # conv
# arch_code = transform_2d(arch_code, [2,2])
# channels = 2
# model = Conv_Net(channels, 1)
# dim = model(arch_code).shape[1]
# model.add_module('classifier', nn.Linear(dim, 1))

Epoch = 3001
true_label = get_label(energy)
t_size = 2000
device = 'cuda'

def data(arch_code_t):
    arch_code_train = arch_code_t[:t_size].to(device)
    energy_train = energy[:t_size].to(device)
    label = get_label(energy_train).to(device)
    p_label = 2 * label - 1
    
    dataset = TensorDataset(arch_code_train, p_label)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    arch_code_test = arch_code_t[t_size:].to(device)
    energy_test = energy[t_size:].to(device)
    test_label = true_label[t_size:].to(device)
    return dataloader, arch_code_train, arch_code_test, energy_test, label, test_label

# mean = energy_test.mean()
# good, bad, good_num, bad_num = 0, 0, 0, 0
# for i in energy_test:
#     if i < mean:
#         good += i
#         good_num += 1
#     else:
#         bad += i
#         bad_num += 1

# print("Ground truth:", good / good_num, bad / bad_num)

def train(model):
    train_loss_list =  []
    if torch.cuda.is_available():
        model.cuda()    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   
    for epoch in range(1, Epoch):
        for x, y in dataloader:
            model.train()
            pred = model(x) 

            loss_e = loss_fn(pred[:, -1], y)             
            train_loss = loss_e #+ 0.1 * loss_s            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()        

        if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(arch_code_train).cpu()
                # error = (pred[:,-1] - energy_train).abs().mean()
                # pred_label = get_label(pred[:, -1])
                pred_label = (pred[:, -1] > 0).float()                
                acc = accuracy_score(pred_label.numpy(), label.cpu().numpy())
                # acc = f1_score(label, pred_label)
                print(epoch, acc)
                train_loss_list.append(acc)
    return train_loss_list

def test(model):
    model.eval()
    with torch.no_grad():
        pred = model(arch_code_test).cpu()
        pred_label = (pred[:, -1] > 0).float()        
        acc = accuracy_score(pred_label.numpy(), test_label.cpu().numpy())
        # acc = f1_score(test_label, pred_label)
        good_num = pred_label.sum()
        bad_num = pred_label.shape[0] - good_num
        mae_good = (energy_test.cpu() * pred_label).sum() / good_num
        mae_bad =  (energy_test.cpu() * (1 - pred_label)).sum() / bad_num
        print("test acc:", acc, mae_good, mae_bad)
        train_loss_list.append(acc)
    print(train_loss_list)
    
    return pred_label

dataloader, arch_code_train, arch_code_test, energy_test, label, test_label = data(arch_code_t)
print("dataset size: ", len(energy))
print("training size: ", t_size)
print("test size: ", len(arch_code_test))

s = time.time()
train_loss_list = train(model)
e = time.time()
print('time: ', e-s)
positive = test(model)

# # mlp, linear
# arch_code_t = change_code(arch_code)
# model = Mlp(21, 8, 2)
# model1 = Linear(21, 2)

# dataloader, arch_code_train, arch_code_test, energy_test, label, test_label = data(arch_code_t)
# train_loss_list = train(model)
# positive_1 = test(model)
# train_loss_list = train(model1)
# positive_2= test(model1)


# print(((positive_2 * positive_1 * positive) * energy_test.cpu()).sum() / (positive_2 * positive_1 * positive).sum())