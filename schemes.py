import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, f1_score
from datasets import MOSIDataLoaders
from FusionModel import QNet
from FusionModel import translator
from Arguments import Arguments
import random
import pickle
import csv, os
import pandas as pd


def get_param_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:', total_num, 'trainable:', trainable_num)


def display(metrics):
    print("\nTest mae: {}".format(metrics['mae']))
    # print("Test correlation: {}".format(metrics['corr']))
    # print("Test multi-class accuracy: {}".format(metrics['multi_acc']))
    # print("Test binary accuracy: {}".format(metrics['bi_acc']))
    # print("Test f1 score: {}".format(metrics['f1']))


def train(model, data_loader, optimizer, criterion, args):
    model.train()
    for data_a, data_v, data_t, target in data_loader:
        data_a, data_v, data_t = data_a.to(args.device), data_v.to(args.device), data_t.to(args.device)
        target = target.to(args.device)
        optimizer.zero_grad()
        output = model(data_a, data_v, data_t)
        loss = criterion(output, target)
        # loss = output[1]
        loss.backward()
        optimizer.step()


def test(model, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data_a, data_v, data_t, target in data_loader:
            data_a, data_v, data_t = data_a.to(args.device), data_v.to(args.device), data_t.to(args.device)
            target = target.to(args.device)
            output = model(data_a, data_v, data_t)
            instant_loss = criterion(output, target).item()
            total_loss += instant_loss
    total_loss /= len(data_loader.dataset)
    return total_loss


def evaluate(model, data_loader, args):
    model.eval()
    metrics = {}
    with torch.no_grad():
        data_a, data_v, data_t, target = next(iter(data_loader))
        data_a, data_v, data_t = data_a.to(args.device), data_v.to(args.device), data_t.to(args.device)
        output = model(data_a, data_v, data_t)
    output = output.cpu().numpy()
    target = target.numpy()
    metrics['mae'] = np.mean(np.absolute(output - target)).item()
    metrics['corr'] = np.corrcoef(output, target)[0][1].item()
    metrics['multi_acc'] = round(sum(np.round(output) == np.round(target)) / float(len(target)), 5).item()
    true_label = (target >= 0)
    pred_label = (output >= 0)
    metrics['bi_acc'] = accuracy_score(true_label, pred_label).item()
    metrics['f1'] = f1_score(true_label, pred_label, average='weighted').item()    
    return metrics


def Scheme(design):
    
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)
    args = Arguments()
    if torch.cuda.is_available() and args.device == 'cuda':
        print("using cuda device")
    else:
        print("using cpu device")
    train_loader, val_loader, test_loader = MOSIDataLoaders(args)
    model = QNet(args, design).to(args.device)
    model.load_state_dict(torch.load('classical_weight'), strict= False)
    criterion = nn.L1Loss(reduction='sum')
    # optimizer = optim.Adam([
    #     {'params': model.ClassicalLayer_a.parameters()},
    #     {'params': model.ClassicalLayer_v.parameters()},
    #     {'params': model.ClassicalLayer_t.parameters()},
    #     {'params': model.ProjLayer_a.parameters()},
    #     {'params': model.ProjLayer_v.parameters()},
    #     {'params': model.ProjLayer_t.parameters()},
    #     {'params': model.QuantumLayer.parameters(), 'lr': args.qlr},
    #     {'params': model.Regressor.parameters()}
    #     ], lr=args.clr)
    optimizer = optim.Adam(model.QuantumLayer.parameters(),lr=args.clr)  #args.qlr = 0.05, clr = 0.005
    train_loss_list, val_loss_list = [], []
    best_val_loss = 10000

    start = time.time()
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, args)
        train_loss = test(model, train_loader, criterion, args)
        train_loss_list.append(train_loss)
        val_loss = test(model, val_loader, criterion, args)
        val_loss_list.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(epoch, train_loss, val_loss, 'saving model')
            best_model = copy.deepcopy(model)
        else:
            print(epoch, train_loss, val_loss)
    end = time.time()
    print("Running time: %s seconds" % (end - start))
    
    metrics = evaluate(best_model, test_loader, args)
    display(metrics)
    report = {'train_loss_list': train_loss_list, 'val_loss_list': val_loss_list,
              'best_val_loss': best_val_loss, 'metrics': metrics}
    ## store classical weights
    # del best_model.QuantumLayer
    # torch.save(best_model.state_dict(), 'classical_weight')
    return best_model, report


if __name__ == '__main__':
    # list = [[1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0]]
    # net = list[1]
    list = []
    df = pd.read_csv('results.csv')
    for index, row in df.iterrows():
        list.append(eval(row['arch_code']))    

    i = 0
    for net in list:
        design = translator(net)
        best_model, report = Scheme(design)
        with open('output.csv', 'a+', newline='') as res:
            writer = csv.writer(res)
            best_val_loss = report['best_val_loss']
            metrics = report['metrics']
            writer.writerow([i, net, best_val_loss, metrics['mae'], metrics['corr'],
                                metrics['multi_acc'], metrics['bi_acc'], metrics['f1']])
        i += 1