import torch.nn as nn
import torch
import pickle
import numpy as np
import time

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(input_size, 1)
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):        #(batch, seq, feature)
        x = x.permute(1, 0, 2)   #(seq, batch, feature)
        out, _ = self.attention(x, x, x)
        out = out.permute(1, 0, 2)
        out = self.linear(out)
        return out   


def positional_encoding(max_len, d_model):
    pos = torch.arange(max_len).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
    angle_rads = pos * angle_rates
    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])
    pos_encoding = torch.cat([sines, cosines], dim=-1)
    return pos_encoding

pos = positional_encoding(35, 3)
# print(pos)

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
    x = x.reshape(-1, 3, 7)
    x_1 = x
    for i in range(repeat[0] -1):
        x_1 = torch.cat((x_1, x), 1)
    x = x_1
    for i in range(repeat[1] -1):
        x = torch.cat((x, x_1), 2)
    return x

with open('data/mosi_dataset', 'rb') as file:
    dataset = pickle.load(file)

arch_code, energy = [], []
for key in dataset:
    arch_code.append(eval(key))
    energy.append(dataset[key])
arch_code = torch.from_numpy(np.asarray(arch_code, dtype=np.float32))
energy =  torch.from_numpy(np.asarray(energy, dtype=np.float32))
arch_code = transform(arch_code, [1, 5])
arch_code = arch_code.transpose(1, 2)
print(arch_code.shape)

model = Attention(3, 1)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)
x = arch_code + pos

s = time.time()
y = model(x)
y = y.flatten(1)
e = time.time()

print("time:", e-s)
print(y.shape)
