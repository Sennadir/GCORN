"""
Main script to run a RGCN on clean graphs.
---
Takes into account the following datasets:
    - Cora
    - CiteSeer
    - PubMed
    - CS
"""

import argparse
import os.path as osp

import copy
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from torch_geometric.utils import add_self_loops, degree, to_dense_adj

from utils import *
import pickle
import time

from gcn import GCN, normalize_tensor_adj

from r_gcn import *

import time
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_exp = 3

    dataset = Planetoid("./data/", args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    data = data.to(device)
    adj_true = to_dense_adj(data.edge_index)[0, :,:]
    norm_adj = normalize_tensor_adj(adj_true)


    l_acc = []
    l_time = []

    l_epoch = []
    l_loss_train = []
    l_acc_train = []
    l_acc_val = []
    #
    for exp in range(num_exp):
        input_time = time.time()
        model_gcn = RGCN(nnodes=norm_adj.shape[0], nfeat=data.x.shape[1],
                            nhid=16, nclass=dataset.num_classes , dropout=0.5,
                                                        device=device).to(device)

        optimizer = torch.optim.Adam(model_gcn.parameters(), lr = args.lr)

        best_val_acc = 0
        for epoch in range(1, args.epochs + 1):
            model_gcn, loss = train(model_gcn, optimizer, data, adj_true)
            train_acc, val_acc, tmp_test_acc = test(model_gcn, data, adj_true)
            if val_acc > best_val_acc:
                best_model = copy.deepcopy(model_gcn)
                best_val_acc = val_acc
                test_acc = tmp_test_acc

            if epoch % 5 == 0:
                l_epoch.append(epoch)
                l_loss_train.append(loss)
                l_acc_train.append(train_acc)
                l_acc_val.append(val_acc)

        output_time = time.time()

        l_acc.append(test_acc)
        l_time.append(output_time-input_time)

    print('Accuracy results for the GCN: {} - {}' .format(np.mean(l_acc) * 100,
                                                        np.std(l_acc) * 100))
    print('Time for the GCN: {} - {}' .format(np.mean(l_time), np.std(l_time)))
