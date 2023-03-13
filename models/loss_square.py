#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


EPS = 1e-8
def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out

def to_sparse(x, mask):
    return x[mask]

def to_dense(x, mask):
    out = x.new_zeros(tuple(mask.size()) + (x.size(-1), ))
    out[mask] = x
    return out

class DGMC(torch.nn.Module):   
    
    r""" The work is based on deep graph consensus.
    Args:
        psi_1 (torch.nn.Module): The first GNN :math:`\Psi_{\theta_1}` which
            takes in node features :obj:`x`, edge connectivity
            :obj:`edge_index`, and optional edge features :obj:`edge_attr` and
            computes node embeddings.
        psi_2 (torch.nn.Module): The second GNN :math:`\Psi_{\theta_2}` which
            takes in node features :obj:`x`, edge connectivity
            :obj:`edge_index`, and optional edge features :obj:`edge_attr` and
            validates for neighborhood consensus.
            :obj:`psi_2` needs to hold the attributes :obj:`in_channels` and
            :obj:`out_channels` which indicates the dimensionality of randomly
            drawn node indicator functions and the output dimensionality of
            :obj:`psi_2`, respectively.
        num_steps (int): Number of consensus iterations.
        k (int, optional): Sparsity parameter. If set to :obj:`-1`, will
            not sparsify initial correspondence rankings. (default: :obj:`-1`)
        detach (bool, optional): If set to :obj:`True`, will detach the
            computation of :math:`\Psi_{\theta_1}` from the current computation
            graph. (default: :obj:`False`)
    """
    def __init__(self, psi_1, psi_2, num_steps, k=-1, detach=False):
        super(DGMC, self).__init__()
        
        self.psi_1 = psi_1
        self.psi_2 = psi_2
        self.num_steps = num_steps
        self.k = k
        self.detach = detach
        self.backend = 'auto'

        self.mlp = Seq(
            Lin(psi_2.out_channels, psi_2.out_channels),
            ReLU(),
			torch.nn.Dropout(0.2),
            Lin(psi_2.out_channels, 1),
			torch.nn.Dropout(0.2)
        )

    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s, pos_world_s, 
                x_t, edge_index_t, edge_attr_t, batch_t, pos_world_t):
        r"""
        Args:
            x_s/t (Tensor): Source graph node features of shape
                :obj:`[batch_size * num_nodes, C_in]`.
            edge_index_s/t (LongTensor): Source graph edge connectivity of shape
                :obj:`[2, num_edges]`.
            edge_attr_s/t (Tensor): Source graph edge features of shape
                :obj:`[num_edges, D]`. Set to :obj:`None` if the GNNs are not
                taking edge features into account.
            batch_s/t (LongTensor): Source graph batch vector of shape
                :obj:`[batch_size * num_nodes]` indicating node to graph
                assignment. Set to :obj:`None` if operating on single graphs.
            pos_world_s/t : positions of nodes in world coordinates
            
        Returns:
            Correspondence matrices without or with GPS mask:`(S_hat, S_LP)`
            variances of correspondences: soft_un.
        """
        h_s = self.psi_1(x_s, edge_index_s, edge_attr_s)
        h_t = self.psi_1(x_t, edge_index_t, edge_attr_t)

        h_s, h_t = (h_s.detach(), h_t.detach()) if self.detach else (h_s, h_t)

        h_s, s_mask = to_dense_batch(h_s, batch_s, fill_value=0)
        h_t, t_mask = to_dense_batch(h_t, batch_t, fill_value=0)

        assert h_s.size(0) == h_t.size(0), 'Encountered unequal batch-sizes'
        (B, N_s, _), N_t = h_s.size(), h_t.size(1)
        R_in, R_out = self.psi_2.in_channels, self.psi_2.out_channels

        if self.k < 1:
            # ------ Dense variant ------ #
            S_hat = h_s @ h_t.transpose(-1, -2)
            S_mask = s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t)
            S_0 = masked_softmax(S_hat, S_mask, dim=-1)[s_mask]

            for _ in range(self.num_steps):
                S = masked_softmax(S_hat, S_mask, dim=-1)
                r_s = torch.randn((B, N_s, R_in), dtype=h_s.dtype,
                                  device=h_s.device)
                r_t = S.transpose(-1, -2) @ r_s

                r_s, r_t = to_sparse(r_s, s_mask), to_sparse(r_t, t_mask)
                o_s = self.psi_2(r_s, edge_index_s, edge_attr_s)
                o_t = self.psi_2(r_t, edge_index_t, edge_attr_t)
                o_s, o_t = to_dense(o_s, s_mask), to_dense(o_t, t_mask)

                D = o_s.view(B, N_s, 1, R_out) - o_t.view(B, 1, N_t, R_out)
                S_hat = S_hat + self.mlp(D).squeeze(-1).masked_fill(~S_mask, 0)
                
            #S_L = masked_softmax(S_hat, S_mask, dim=-1)
            
            # ***added for GPS mask
            p_s, _ = to_dense_batch(pos_world_s, batch_s)
            p_t, _ = to_dense_batch(pos_world_t, batch_t)
            
            T = torch.div(1, torch.cdist(p_s, p_t, p=2))
            #S_gps = masked_softmax(T, S_mask, dim=-1)
            
            S_LP = torch.add(S_hat, T)
            S_LP = masked_softmax(S_LP, S_mask, dim=-1)[s_mask]

            # S_LP has padding on dim 1, so we cannot simply calculate std on dim 1
            soft_un = torch.zeros(S_LP.size(0), device=S_LP.device)
            for i in range(len(soft_un)):
                soft_un[i] = S_LP[i][t_mask[batch_s[i]]].std()

            return S_hat, S_LP, soft_un
    

    def loss(self, S, y):

        index = y[0]
        gt = y[1]
        truth = torch.zeros(S.shape).to(S.device)
        for c in (gt != 100).nonzero():
            truth[index[c], gt[c]] = 1
        loss = torch.mean(torch.square(S - truth))
        
        return loss


    def prediction(self, S, co_index, batch_s, batch_t, pos_world_s, pos_world_t):
        p_s, s_mask = to_dense_batch(pos_world_s, batch_s)
        p_t, t_mask = to_dense_batch(pos_world_t, batch_t)
        S_dense, _ = to_dense_batch(S, batch_s)
        (B, N_s), N_t = s_mask.size(), t_mask.size(1)
        S_mask = s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t)

        T = torch.div(1, torch.cdist(p_s, p_t, p=2))
        T = T.masked_fill(~S_mask, 0)
        Post_gps = F.normalize(T, p=1, dim=-1)
        S_dense = torch.mul(S_dense, Post_gps)
        
        pred = np.array([])
        # Hungarian
        for b in range(B):
            S_single = S_dense[b][S_mask[b].nonzero(as_tuple=True)].view(s_mask[b].sum(), t_mask[b].sum())
            pred_single = linear_sum_assignment(-S_single.cpu().detach().numpy())
            pred = np.append(pred, pred_single[1])
        return torch.from_numpy(pred).to(S.device)
        

    def acc(self, pred, co_index, y, reduction='mean'):

        assert reduction in ['mean', 'sum']
        gt=y[1]
        #pred=S[y[0]].argmax(dim=-1);
        pred_=pred[co_index]
        gt_=gt[co_index];
        correct = (pred_ == gt_).sum().item()
        
        if pred_.size(0)>0:
            res=correct / pred_.size(0)
        else:
            res=0
                
        return res 
    
    def recall(self, pred, co_index, y,  reduction='mean'):

        gt=y[1]
        pred_=pred[co_index]
        gt_=gt[co_index]
        correct = (pred_ == gt_).sum().item()
        co_gt=(gt!=100).nonzero()

        if co_gt.size(0)>0:
            res=correct / co_gt.size(0)
        else:
            res=0             
        return res

    def __repr__(self):
        return ('{}(\n'
                '    psi_1={},\n'
                '    psi_2={},\n'
                '    num_steps={}, k={}\n)').format(self.__class__.__name__,
                                                    self.psi_1, self.psi_2,
                                                    self.num_steps, self.k)
