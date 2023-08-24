import random
import pickle
import csv
import numpy as np
import torch
from torch import nn, flip
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
from sklearn.metrics import accuracy_score, f1_score
from Network import Linear, Mlp, Conv_Net, Attention, get_label,change_code, transform_2d, transform_attention, positional_encoding

torch.cuda.is_available = lambda : False

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

# # attention
# arch_code = transform_attention(arch_code, [1, 5])   # 5 layers
# pos = positional_encoding(35, 3)
# arch_code = arch_code.transpose(1, 2) + pos
# model = Attention(3, 1)

# # linear
# arch_code = change_code(arch_code)
# model = Linear(21, 1)

# mlp
arch_code = change_code(arch_code)
model = Mlp(21, 6, 1)

# # conv
# arch_code = transform_2d(arch_code, [2,2])
# channels = 2
# model = Conv_Net(channels, 1)
# dim = model(arch_code).shape[1]
# model.add_module('classifier', nn.Linear(dim, 1))

Epoch = 3001

true_label = get_label(energy)
t_size = 4000
arch_code_train = arch_code[:t_size]
energy_train = energy[:t_size]
label = get_label(energy_train)

if torch.cuda.is_available():
    arch_code_train = arch_code_train.cuda()
    energy_train = energy_train.cuda()
    label = label.cuda()

dataset = TensorDataset(arch_code_train, label)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

arch_code_test = arch_code[t_size:]
energy_test = energy[t_size:]
test_label = true_label[t_size:]

mean = energy_test.mean()
good, bad, good_num, bad_num = 0, 0, 0, 0
for i in energy_test:
    if i < mean:
        good += i
        good_num += 1
    else:
        bad += i
        bad_num += 1

print("dataset size: ", len(energy))
print("training size: ", len(energy_train))
print("test size: ", len(arch_code_test))
print("Ground truth:", good / good_num, bad / bad_num)

if torch.cuda.is_available():
    arch_code_test = arch_code_test.cuda()
    # energy_test = energy_test.cuda()
    test_label = test_label.cuda()

# for channel in range(1, 10, 2):

if torch.cuda.is_available():
    model.cuda()    
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loss_list, test_loss_list = [], []
s = time.time()
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
            pred_label = (pred[:, -1] > 0.5).float()
            label = label.cpu()
            acc = accuracy_score(pred_label.numpy(), label.numpy())
            # acc = f1_score(label, pred_label)
            print(epoch, acc)
            train_loss_list.append(acc)
    
model.eval()
with torch.no_grad():
    pred = model(arch_code_test).cpu()
    pred_label = (pred[:, -1] > 0.5).float()
    test_label = test_label.cpu()
    acc = accuracy_score(pred_label.numpy(), test_label.numpy())
    # acc = f1_score(test_label, pred_label)
    good_num = pred_label.sum()
    bad_num = pred_label.shape[0] - good_num
    mae_good = (energy_test * pred_label).sum() / good_num
    mae_bad =  (energy_test * (1 - pred_label)).sum() / bad_num
    print("test acc:", acc, mae_good, mae_bad)
    train_loss_list.append(acc)
e = time.time()
print('time: ', e-s)
print(train_loss_list)

# plt.plot(range(len(train_loss_list)), train_loss_list, 'ro-')
# plt.title('min test loss')
# plt.xlabel('hidden dim')
# plt.ylabel('test loss')
# plt.show()
