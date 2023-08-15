import pennylane as qml
import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from Arguments import Arguments
import warnings
import matplotlib.cbook
from FusionModel import translator, quantum_net
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

dir_path = os.path.dirname(os.path.realpath(__file__))
files = os.listdir(dir_path)
# dataset_file = os.path.join(dir_path, 'mosi_dataset')

args = Arguments()
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

# csv_reader=csv.reader(open("result.csv"))
lst,res, mae=[], [], []
# for row in csv_reader:
#     lst.append(row[1])
# lst.pop(0)

df = pd.read_excel('result.xlsx', sheet_name='Sheet1')
for index, row in df.iterrows():
    lst.append(eval(row['arch_code']))
    mae.append(row['test_mae'])

ChooseNum=len(lst)

code = change_code(torch.tensor(lst)).numpy()

for i in range(ChooseNum):
    # net = [0, 0, 2, 2, 1, 0, 2, 0, 0, 1, 1, 0, 1, 1, 0, 6, 0, 1, 4, 3, 2, 2]
    design = translator(lst[i])
    input = nn.Parameter(torch.rand(args.n_qubits * 3))
    # weight = nn.Parameter(torch.rand(design['layer_repe'] * args.n_qubits * 2))
    q_params_rot = nn.Parameter(torch.rand(design['layer_repe'] * args.n_qubits))
    q_params_enta = nn.Parameter(torch.rand(design['layer_repe'] * (args.n_qubits-1)))
    fig, _ = qml.draw_mpl(quantum_net, decimals=1, style="black_white", fontsize="x-small")(input, q_params_rot, q_params_enta, design = design)
    fig.suptitle(str(code[i]) + ': ' + str(mae[i]), fontsize="xx-large")   
    plt.savefig('results_'+str(i)+'.jpg')    
    # fig.show()
    # plt.clf()
# print(code)