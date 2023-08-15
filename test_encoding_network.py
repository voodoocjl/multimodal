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

torch.cuda.is_available = lambda : False
# torch.set_num_threads(4)

# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1000),
            # nn.Dropout(p=0.5),
            nn.Sigmoid(),
            nn.Linear(1000, hidden_dim),
            # nn.Dropout(p=0.5),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)            
            )
        
    def forward(self, x):
        y = self.network(x)
        # y[:,-1] = torch.sigmoid(y[:,-1])
        # y = torch.round(y)
        return y

class Enco_Conv_Net(nn.Module):
    def __init__(self, n_channels, output_dim):
        super(Enco_Conv_Net, self).__init__()
        hidden_chanels = 64
        self.features_2x4 = nn.Sequential(
            nn.Conv2d(1, hidden_chanels, kernel_size= (2, 4)),
            nn.ReLU(),
            # nn.MaxPool2d(2,2)            
            )
        self.dropout = nn.Dropout2d(p=0.5)
        self.pool1d = nn.MaxPool1d(2, 2)
        self.features_2x2 = nn.Sequential(
            nn.Conv2d(hidden_chanels, n_channels, kernel_size = 2),
            nn.ReLU(),
            # nn.MaxPool1d(3, 2)        
            )
        self.classifier = nn.Linear(n_channels * 5, output_dim)

    def forward(self, x):
        # x = transform(x)
        x1 = self.features_2x4(x)
        x1 = self.dropout(x1)
        x2 = self.features_2x2(x1)
        x2 = self.pool1d(x2.squeeze(2))
        # x1 = x1.flatten(1)        
        x2 = x2.flatten(1)
        # x_ = torch.cat((x1, x2), 1)
        y = self.classifier(x2)
        y[:,-1] = torch.sigmoid(y[:,-1])
        return y

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

def get_label(energy):
    label = energy.clone()
    for i in range(energy.shape[0]): 
        label[i] = energy[i] < energy.mean()
    return label

def change_code(x):
    """x- torch.Tensor, 
    shape: (batch_size, arch_code_len), 
    dtype: torch.float32"""
    pos_dict = {'00': 3, '01': 4, '10': 5, '11': 6}
    x_ = torch.Tensor()
    for elem in x:
        q = elem[0:7]
        c = torch.cat((elem[7:13], torch.zeros(1,dtype=torch.float32)))
        p = elem[13:].int().tolist()
        p_ = torch.zeros(7, dtype=torch.float32)
        for i in range(3):
            p_[i] = pos_dict[str(p[2*i]) + str(p[2*i+1])]
        for j in range(3, 6):
            p_[j] = j + 1
        elem_ = torch.cat((q, c, p_))
        x_ = torch.cat((x_, elem_.unsqueeze(0)))
    return x_

def transform(x, repeat = [1, 1]):
    x = change_code(x)    
    # single_type = x[:, :7].unsqueeze(1)
    # entangle_type = x[:, 7:14].unsqueeze(1)
    # entangle_position = x[:, 14:21].unsqueeze(1)
    # x = torch.cat((single_type, entangle_type, entangle_position), 1)
    x = x.reshape(-1, 3, 7)
    x_flip = flip(x, dims = [2])
    x = torch.cat((x_flip, x), 2)
    x_1 = x
    for i in range(repeat[0] -1):
        x_1 = torch.cat((x_1, x), 1)
    x = x_1
    for i in range(repeat[1] -1):
        x = torch.cat((x, x_1), 2)
    return x.unsqueeze(1)

with open('data/mosi_dataset', 'rb') as file:
    dataset = pickle.load(file)

arch_code, energy = [], []
for key in dataset:
    arch_code.append(eval(key))
    energy.append(dataset[key])
arch_code = torch.from_numpy(np.asarray(arch_code, dtype=np.float32))
energy =  torch.from_numpy(np.asarray(energy, dtype=np.float32))
arch_code = transform(arch_code)

true_label = get_label(energy)
t_size = 10000
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
test_label = true_label[t_size:]

if torch.cuda.is_available():
    arch_code_test = arch_code_test.cuda()
    # energy_test = energy_test.cuda()
    test_label = test_label.cuda()

print("dataset size: ", len(energy))
print("training size: ", len(energy_train))
print("test size: ", len(arch_code_test))

for channel in range(1, 10, 2):
    model = Enco_Conv_Net(channel, 1)
    # model = Encoder(19, 100, 1)
    if torch.cuda.is_available():
        model.cuda()    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_loss_list, test_loss_list = [], []
    s = time.time()
    for epoch in range(1, 3001):
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
        print("test acc:", acc)
        # train_loss_list.append(acc)
    e = time.time()
    print('time: ', e-s)

    model.eval()
    with torch.no_grad():
        pred = model(arch_code_test).cpu()      
        pred_label = (pred[:, -1] > 0.5).float()
        test_label = test_label.cpu()
        acc = accuracy_score(pred_label.numpy(), test_label.numpy())
        # acc = f1_score(test_label, pred_label)
        print("test acc:", acc)
        train_loss_list.append(acc)
    print(train_loss_list)

plt.plot(range(len(train_loss_list)), train_loss_list, 'ro-')
plt.title('min test loss')
plt.xlabel('hidden dim')
plt.ylabel('test loss')
plt.show()
