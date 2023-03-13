#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch.nn import Linear as Lin


class TransGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_layers, num_head,
                 cat=True, lin=True, dropout=0.0):
        super(TransGNN, self).__init__()

        self.in_channels = in_channels
        self.num_edge_dim = dim
        self.num_layers = num_layers
        self.cat = cat
        self.lin = lin
        self.dropout = dropout
        self.heads = num_head

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = TransformerConv(in_channels, out_channels, heads=self.heads, concat=True, 
                           dropout=self.dropout, edge_dim=self.num_edge_dim)
            self.convs.append(conv)
            in_channels = out_channels * self.heads
        # for the last layer, aggregate the multi-head attention instead of concatenate
        conv = TransformerConv(in_channels, out_channels, heads=self.heads, concat=False, 
                       dropout=self.dropout, edge_dim=self.num_edge_dim)
        self.convs.append(conv)

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, edge_attr, *args):
        """"""
        xs = [x]

        for conv in self.convs:
            xs += [F.relu(conv(xs[-1], edge_index, edge_attr))]
            #xs += [F.dropout(F.relu(conv(xs[-1], edge_index, edge_attr)), 
            #                 p=self.dropout, training=self.training)]
        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, dim={}, num_layers={}, cat={}, lin={}, '
                'dropout={})').format(self.__class__.__name__,
                                      self.in_channels, self.out_channels,
                                      self.dim, self.num_layers, self.cat,
                                      self.lin, self.dropout)

