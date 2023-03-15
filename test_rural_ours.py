#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
sys.path.append('../')
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import pickle

import torch
from torch_geometric.loader import DataLoader


from utils.data_preprocess import traindata, testdata, PairDataset
from models.kernel_trans import TransGNN
from models.loss_square import DGMC
from utils.draw_matching import draw_matching

model_path = './checkpoint/'
test_method = 'DMGM'
save_path = './results'

start = 69467 #60260 is the start idex for test dataset
num_testdata = 1#5462 is the number of data instances in the test dataset
num_totaldata = start + num_testdata


need_to_train = False
need_to_draw = False

parser = argparse.ArgumentParser()
parser.add_argument('--isotropic', action='store_true')
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--rnd_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=500)    # original: 500
parser.add_argument('--epochs', type=int, default=60)  # original: 150
#parser.add_argument('--test_samples', type=int, default=2000)
args = parser.parse_args()
if need_to_train:
    args.batch_size = 500 # 1 for debug, 500 for normal train
if need_to_draw:
    args.batch_size = 1


in_path_s = os.path.join('data/testdata/proc_s')
in_path_t = os.path.join('data/testdata/proc_t')
out_path_s = os.path.join('data/testdata/proc_s')
out_path_t = os.path.join('data/testdata/proc_t')


gps_noise_datasets_s = [testdata(in_path_s, out_path_s, cat) 
                   for cat in list(map("{:05d}".format, range(start, num_totaldata)))]
gps_noise_datasets_t = [testdata(in_path_t, out_path_t, cat) 
                   for cat in list(map("{:05d}".format, range(start, num_totaldata)))]
gps_noise_datasets = [PairDataset(test_s, test_t, sample=False) 
                      for test_s, test_t in zip(gps_noise_datasets_s, gps_noise_datasets_t)]
test_datasets = torch.utils.data.ConcatDataset(gps_noise_datasets)

num_edge_features = test_datasets.datasets[0].dataset_s.num_edge_features
num_node_features = test_datasets.datasets[0].dataset_s.num_node_features

device = 'cuda' if torch.cuda.is_available() else 'cpu'
psi_1 = TransGNN(num_node_features, args.dim, num_edge_features, args.num_layers,
               num_head=4, cat=False, dropout=0.5)
psi_2 = TransGNN(args.rnd_dim, args.rnd_dim, num_edge_features, args.num_layers, 
               num_head=4, cat=False, dropout=0.5)

model = DGMC(psi_1, psi_2, num_steps=args.num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def generate_y(y_col):
    y_row = torch.arange(y_col.size(0), device=device)
    return torch.stack([y_row, y_col], dim=0)


@torch.no_grad()
def test(dataset, epoch=None):
    model.eval()

    test_loader = DataLoader(dataset, args.batch_size, shuffle=False,
                        follow_batch=['x_s', 'x_t'])

    precision = 0
    recall = 0
    rec = np.zeros(51)
    correct = np.zeros(51)
           
    for data in test_loader:
        # data is one batch of graph inputs
        data = data.to(device)
        
        S_0, S_L, soft_un = model(
                data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch, 
                data.pos_world_s, data.x_t, data.edge_index_t, data.edge_attr_t, 
                data.x_t_batch, data.pos_world_t)

        y = generate_y(data.y)          
        
        for it, thresh in enumerate(np.linspace(0,0.5,51)):
            #thresh=0.25
            co_index_soft=soft_un>=thresh
            pred = model.prediction(S_L, co_index_soft, data.x_s_batch, data.x_t_batch, data.pos_world_s, data.pos_world_t)
            #correct[it] += model.acc(S_L.argmax(dim=-1), co_index_soft,y, reduction='mean')
            correct[it] += model.acc(pred, co_index_soft, y, reduction='mean') 
            rec[it] += model.recall(pred, co_index_soft, y, reduction='mean')
        
    precision = correct / len(test_loader)
    recall = rec / len(test_loader)
    f1 = np.divide(2 * np.multiply(precision, recall), np.add(precision, recall))
    max_idx = np.nanargmax(f1)
    result_filename = os.path.join(save_path, 'result_test')

    with open(result_filename, 'a') as file:
        file.write("precision:\n")
        for p in precision:
            file.write(repr(p) + ' ')
        file.write('\nrecall:\n')
        for r in recall:
            file.write(repr(r) + ' ')
        file.write('\nf1:\n')
        for f in f1:
            file.write(repr(f) + ' ')
        file.write('\nprecision of the max f1 score: ' + repr(precision[max_idx]))
        file.write('\nrecall of the max f1 score: ' + repr(recall[max_idx]))
        file.write('\nmax f1 score: ' + repr(np.nanmax(f1)))
        file.write('\nthreshold for max f1 score: ' + repr(max_idx * 0.01) + '\n')
    return precision, recall, f1, max_idx



@torch.no_grad()
def test_PR(dataset, threshold, epoch=None):
    model.eval()

    test_loader = DataLoader(dataset, args.batch_size, shuffle=False,
                        follow_batch=['x_s', 'x_t'])
    
    pred = []
    masked_G = []
    for data in test_loader:
        # data is one batch of graph inputs
        data = data.to(device)
        
        S_0, S_L, soft_un = model(
                data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch, 
                data.pos_world_s, data.x_t, data.edge_index_t, data.edge_attr_t, 
                data.x_t_batch, data.pos_world_t)

        y = generate_y(data.y)          
        co_index = soft_un>=threshold
        index = y[0]
        gt = y[1]
        probs = S_L
        
        for i in range(probs.size(0)):
            if co_index[i]==True:
                continue
            else:
                probs[i,:]=0;
        
        pred.extend(probs.view(-1).cpu().numpy())
        G = torch.zeros(probs.size());
        for i in range(len(index)):
            if gt[i]!=100:
                G[index[i],gt[i]] = 1
        
        masked_G.extend(G.view(-1).cpu().numpy())
    
    precision, recall, thresholds = precision_recall_curve(masked_G, pred)
    plt.plot(recall, precision)
    plt.savefig(os.path.join(save_path, 'pr.png'))
        
    savefile = open(os.path.join(save_path, 'prdata.pk'), 'wb')
    pickle.dump({"precision": precision, "recall": recall, "thresholds": thresholds}, savefile)
    savefile.close()
    
    #res = {"truth": masked_G, "score": pred}
    #savemat(os.path.join(save_path, method, 'plot', 'pr.mat'), res)


if (not need_to_train) and (not need_to_draw):
    # load model
    model.load_state_dict(torch.load(os.path.join(model_path, test_method))) 

    precision, recall, f1, max_idx = test(test_datasets)
    test_PR(test_datasets, max_idx * 0.01)
    


