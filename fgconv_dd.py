#!/usr/bin/env python
# coding: utf-8

# PyTorch for Framelet Graph Convolution (FGConv)
# Yu Guang Wang (UNSW) & Xuebin Zheng (USYD), April 2020

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from torch_geometric.utils import add_remaining_self_loops, to_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_scatter import scatter_add
import scipy
from scipy import sparse
import scipy.sparse as sp
import scipy.io as sio
from sklearn.cluster import SpectralClustering
import warnings
import argparse
import os
import os.path as osp
import matplotlib.pyplot as plt
import copy
import math
warnings.filterwarnings("ignore")  # ignore all warnings
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)


def tree_idx2(treeG, k1, J1, J2):
    """
    finds all indices at level J2 of the k1-th node of level J1 for J2 < J1
    here the graph at higher level is coarser in the chain

    :param treeG: a list of graphs from the tree
    :param k1: k1-th node at the level J1
    :param J1: a coarser level
    :param J2: a finer level
    :return: indices of all nodes whose parent at level J1 is k1
    """
    g = treeG[J1]['clusters'][k1]
    if J1 > J2 + 1:
        for j in np.arange(J2 + 1, J1)[::-1]:
            g1 = []
            for i in range(len(g)):
                g1 = np.array(np.append(g1, treeG[j]['clusters'][g[i]]), dtype=int)
            g = g1
    y = g
    return y


def addweight1(treeG):
    """
    adds weights for each level of the chain treeG

    :param treeG: a chain with clusters at each level
    :return: the chain at level has weight property treeG[j]['w']
    """
    J = len(treeG)
    N = len(treeG[0]['clusters'])
    W = np.zeros((J, N))
    # equal weights for the first level (i.e. the finest level)
    W[0, :] = 1.0
    for j in range(J):
        clusterj = treeG[j]['clusters']
        Nj = len(clusterj)
        w = np.zeros(Nj)
        for k in range(Nj):
            cj_k = len(clusterj[k])
            w[k] = 1.0 / np.sqrt(cj_k)
        treeG[j]['w'] = w
        # compute weights W_k^{(j)}
        if j > 0:
            wj = w
            wj1 = np.zeros(N)
            # expand the weights from level j to 0
            for k in range(Nj):
                idx = tree_idx2(treeG, k, j, 0)
                wj1[idx] = wj[k]
                W[j, :] = W[j - 1, :] * wj1
                # find the indices of level 0 who have the same parent
                # which is the k-th vertex of level j
    treeG[-1]['W'] = W
    return treeG


def HaarGOB(treeG):
    """
    HaarGOB generates Haar Global Orthonormal Basis for a chain or tree treeG

    INPUTS:
    treeG  - chain or tree for a graph
    OUTPUTS:
    treeG1 - the updated chain treeG which is added for each level a Haar
            orthonormal basis and for the bottom level the basis is the
            Haar Global Orthonormal Basis
    """
    # number of level of the chain (or tree)
    Ntr = len(treeG)
    # reorder chain (optional)
    # reordering each level so that in each level the nodes are in the
    # descent order of degrees
    # compute u_l^c for level J_0 (top level)
    clusterJ0 = treeG[Ntr - 1]['clusters']
    N0 = len(clusterJ0)
    # generate indicator function on G^c
    chic = np.identity(N0)
    U = np.zeros((N0, N0))
    uc = [None] * N0
    uc[0] = 1 / np.sqrt(N0, dtype=np.float64) * np.ones(N0)
    U[0, :] = uc[0]
    for l in np.arange(1, N0):
        uc[l] = np.sqrt((N0 - l) / (N0 - l + 1)) * (chic[l - 1, :] - 1 / (N0 - l) * np.sum(chic[l:, :], axis=0))
        U[l, :] = uc[l]
    #    u = copy.deepcopy(uc)
    treeG[Ntr - 1]['u'] = uc
    treeG[Ntr - 1]['U'] = U
    # compute the next level orthonormal basis ulk and stored into u
    for j_tr in np.arange(0, Ntr - 1)[::-1]:
        N1 = len(treeG[j_tr]['clusters'])
        u = [None] * N1
        U = np.zeros((N1, N1))
        i = N0
        for l in range(N0):
            clusterl = treeG[j_tr + 1]['clusters'][l]
            kl = len(clusterl)
            # for k==1
            ucl = uc[l]
            ul1 = np.zeros(N1)
            for j in range(N0):
                idxj = treeG[j_tr + 1]['clusters'][j]
                ul1[idxj] = ucl[j] / np.sqrt(len(idxj))
            u[l] = ul1
            U[l, :] = u[l]
            if kl > 1:
                chil = np.zeros((kl, N1))
                for k in range(kl):
                    idxl = treeG[j_tr + 1]['clusters'][l]
                    chil[k, idxl[k]] = 1

                for k in np.arange(1, kl):
                    i = i + 1
                    ulk = np.sqrt((kl - k) / (kl - k + 1)) * (
                                chil[k - 1, :] - 1 / (kl - k) * np.sum(chil[k:, :], axis=0))
                    u[i - 1] = ulk
                    U[i - 1, :] = u[i - 1]
        treeG[j_tr]['u'] = u
        treeG[j_tr]['U'] = U
        # update uc and N0
        #        uc = copy.deepcopy(u)
        uc = u
        N0 = N1
    return treeG


def SC_tree(adjacency_matrix, levels, ratio=0.5):
    """
    Coarsen a graph multiple times using the SpectralClustering algorithm.
    INPUT
        W: symmetric sparse weight (adjacency) matrix
        levels: the number of coarsened graphs
    OUTPUT
        treeG with three objects:
        - IDX (presenting parents) at each level
        - clusters at each level
        - coarsened adjacency matrix at each level
    """
    parents = []
    adj_list = []
    W = adjacency_matrix
    N_start, N_start = W.shape
    adj_list.append(sp.csr_matrix(W))
    for k in range(levels - 1):
        # levels==0 stands for the original graph
        # PAIR THE VERTICES AND CONSTRUCT THE ROOT VECTOR
        idx_row, idx_col, val = scipy.sparse.find(W)
        perm = np.argsort(idx_row)
        rr = idx_row[perm]
        cc = idx_col[perm]
        vv = val[perm]
        """
        Using SpectralClustering at each level
        """
        sc = SpectralClustering(n_clusters=int(W.shape[0] * ratio) + 1, affinity='precomputed', n_init=10)
        cluster_id = sc.fit(W).labels_
        parents.append(cluster_id)
        # COMPUTE THE EDGES WEIGHTS FOR THE NEW GRAPH
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1
        # CSR is more appropriate: row,val pairs appear multiple times
        W = sp.csr_matrix((nvv, (nrr, ncc)), shape=(Nnew, Nnew))
        adj_list.append(W)  # saving the coarsened adj

    """
    Constructing the tree 
    """
    treeG = list(np.arange(0, levels, 1))
    # Obtain the clusters and IDX, saved in treeG
    for i in range(levels):
        if i == 0:  # special case for the base level, corresponding to the original graph
            parents_ini = np.arange(0, N_start, 1)
            idx_sort = np.argsort(parents_ini)
            sorted_records_array = parents_ini[idx_sort]
            vals, idx_start, count = np.unique(sorted_records_array, return_counts=True,
                                               return_index=True)
            cluster_1 = np.split(idx_sort, idx_start[1:])
            treeG[i] = {'IDX': np.arange(0, N_start, 1), 'clusters': cluster_1, 'adj': adj_list[0]}
        else:  # the second level to the top level
            idx_sort = np.argsort(parents[i - 1])
            sorted_records_array = parents[i - 1][idx_sort]
            vals, idx_start, count = np.unique(sorted_records_array, return_counts=True,
                                               return_index=True)
            cluster_temp = np.split(idx_sort, idx_start[1:])
            treeG[i] = {'IDX': parents[i - 1], 'clusters': cluster_temp, 'adj': adj_list[i]}
    return treeG


def aDFT_gL(c, treeG):
    return torch.matmul(c, torch.tensor(treeG[0]['U'], dtype=torch.float64).to(device))


def DFT_gL(f, treeG):
    return torch.matmul(f, torch.tensor(treeG[0]['U'].T, dtype=torch.float64).to(device))


def up_f_hat(f_hat, Nj):
    """
    upsample f_hat vector to level j with length Nj.
    [PyTorch-enabled Function]

    :param f_hat: a matrix of size (d,nj1), d-dim feature, nj1 < nj, torch tensor format
    :param Nj: length Nj, must be integer
    :return: a vector which components are those of f_hat from 1 to nj,
                from 1:nj1, f_hat_down is f_hat, from nj1 + 1:nj, f_hat_down is zeros,
                torch tensor format
    """
    d, Nj1 = f_hat.shape
    f_hat_up = torch.zeros(d, Nj)
    f_hat_up[:, :Nj1] = f_hat
    return f_hat_up


def dwn_f_hat(f_hat, nj1):
    """
    downsample f_hat vector to level j - 1 with length Nj1.
    [PyTorch-enabled Function]

    :param f_hat: a matrix of size (d,j1), d-dim feature, nj > nj1, torch tensor format
    :param nj1: length nj1, must be integer
    :return: a vector which components are those of f_hat from 1 to nj1,
                torch tensor format
    """
    f_hat_dwn = f_hat[:, :nj1]
    return f_hat_dwn


def nu(t):
    """
    elementary function nu(t) satisfies
        1. nu(t) + nu(1 - t) = 1
        2. nu(t) = 0 for t <= 0
        3. nu(1) = 1 for t >= 1

    :param t: data points on the real line R arranged in numpy array
    :return: function values nu(t) in numpy array format
    """
    y = ((t >= 0) & (t < 1)) * t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3) + (t >= 1)
    return y


def hmask(x, cL, epsL, cR, epsR):
    """
    generate a filter y(x) satisfies
        1. y(x) is supported on [cL - epsL, cR + epsR]
        2. y(x) = 1 on [cL + epsL, cR - epsR]

    :param x: data points on the real line R arranged in numpy array
    :param cL: control point, essential support
    :param epsL: shape parameter
    :param cR: control point, essential support
    :param epsR: shape parameter
    :return: y(x) supported on [cL - epsL, cR + epsR] in numpy array format
    """
    y = np.float64((x <= cR - epsR) * (x >= cL + epsL))
    y += np.sin(np.pi / 2 * nu((x - cL + epsL) / epsL / 2)) * (x < cL + epsL)
    y += np.cos(np.pi / 2 * nu((x - cR + epsR) / epsR / 2)) * (x > cR - epsR)
    return y


def filter_bank_1high(t, Nj, Nj_1, ac=2.0):
    # a_hat
    a_cR = (1 + Nj_1) / ac
    a_epsR = Nj_1 - a_cR
    a_cL = -a_cR
    a_epsL = a_epsR
    # b_hat_1
    b1_cL = a_cR
    b1_epsL = a_epsR
    b1_cR = 2 * Nj
    b1_epsR = Nj / 4

    # supp(ha) = [0, 1 / 4]
    ha = hmask(t, a_cL, a_epsL, a_cR, a_epsR)
    # supp(hb1) = [1 / 8, 1 / 2]
    hb1 = hmask(t, b1_cL, b1_epsL, b1_cR, b1_epsR)

    return ha, hb1


def filter_bank_2high(t, Nj, Nj_1, ac=2.0, bc=2.0):
    """
    computes the filter bank for control points N_j, Nj_1 given the variable t

    :param t: data points on the real line R arranged in numpy array
    :param Nj: control point, Nj > Nj_1, integer
    :param Nj_1: control point, Nj > Nj_1, integer
    :param ac: between (1, 2]. Default 2.0
    :param bc: bc < 2. Default 2.0
    :return: (ha, hb1, hb2) low-pass filter ha and high-pass filters hb1 and hb2 at t,
                all in numpy array format
    """
    # a_hat
    a_cR = (1 + Nj_1) / ac
    a_epsR = Nj_1 - a_cR
    a_cL = -a_cR
    a_epsL = a_epsR
    # b_hat_1
    b1_cL = a_cR
    b1_epsL = a_epsR
    b1_cR = (Nj_1 + Nj) / bc
    b1_epsR = Nj - b1_cR
    # b_hat_2
    b2_cL = b1_cR
    b2_epsL = b1_epsR
    b2_cR = 2 * Nj
    b2_epsR = 1

    # supp(ha) = [0, 1 / 4]
    ha = hmask(t, a_cL, a_epsL, a_cR, a_epsR)
    # supp(hb1) = [1 / 8, 1 / 2]
    hb1 = hmask(t, b1_cL, b1_epsL, b1_cR, b1_epsR)
    # supp(hb2) = [1 / 4, 1 / 2]
    hb2 = hmask(t, b2_cL, b2_epsL, b2_cR, b2_epsR)

    return ha, hb1, hb2


def filter_bank_3high(t, Nj, Nj_1, ac=0.25, bc1=0.25, bc2=0.25):
    """
    Computes the filter bank for control points Nj, Nj_1 given the variable t

    :param t: data points on the real line R arranged in numpy array
    :param Nj: control point, Nj > Nj_1, integer
    :param Nj_1: control point, Nj > Nj_1, integer
    :param ac: between (0, 1) which affect skewness of shapes of ha and hbj. Default 0.25
    :param bc1: between (0, 1) which affect skewness of shapes of ha and hbj. Default 0.25
    :param bc2: between (0, 1) which affect skewness of shapes of ha and hbj. Default 0.25
    :return: (ha, hb1, hb2, hb3) low-pass filter ha and high-pass filters hb1, hb2 and hb3 at t,
                all in numpy array format
    """
    # a_hat
    Na = Nj_1  # touch down point of a to the right
    a_cR = 0.5 * Na + 0.5 * ac * Na
    a_epsR = Na - a_cR
    a_cL = -a_cR
    a_epsL = a_epsR

    # b_hat_1
    beta1 = 0.3  # split [Nj_1, Nj] to about 3 equal pieces
    Nb1 = Na + beta1 * (Nj - Nj_1)  # touch down point of b to the right
    b1_cL = a_cR
    b1_epsL = a_epsR
    b1_cR = 0.5 * (Na + Nb1) + 0.5 * bc1 * (Nb1 - Na)  # default bc1 = 0.25
    b1_epsR = Nb1 - b1_cR

    # b_hat_2
    beta2 = 0.8
    Nb2 = Na + beta2 * (Nj - Nj_1)  # touch down point of a to the right
    b2_cL = b1_cR
    b2_epsL = b1_epsR
    b2_cR = 0.5 * (Nb1 + Nb2) + 0.5 * bc2 * (Nb2 - Nb1)
    b2_epsR = Nb2 - b2_cR

    # b_hat_3
    b3_cL = b2_cR
    b3_epsL = b2_epsR
    b3_cR = 2 * Nj
    b3_epsR = 1

    # supp(ha) = [0, 1 / 4]
    ha = hmask(t, a_cL, a_epsL, a_cR, a_epsR)
    # supp(hb1) = [1 / 8, 1 / 2]
    hb1 = hmask(t, b1_cL, b1_epsL, b1_cR, b1_epsR)
    # supp(hb2) = [1 / 4, 1 / 2]
    hb2 = hmask(t, b2_cL, b2_epsL, b2_cR, b2_epsR)
    # supp(hb2) = [1 / 4, 1 / 2]
    hb3 = hmask(t, b3_cL, b3_epsL, b3_cR, b3_epsR)

    return ha, hb1, hb2, hb3


def FGT_Decomp_tensor(f, treeG, max_tree_nodes, num_high_pass=3):
    """
    This function performs FGT decomposition which decomposes a graph signal (in time domain) into
    a set of high-pass and low-pass coefficients (in time domain). The length of each high-pass set
    and low-pass set is dependent on the generated tree structure.
    [PyTorch-enabled Function]

    :param max_tree_nodes: max of nodes.
    :param f: a matrix of graph signal with size [d, len(treeG[0]['clusters'])], d is the feature dimension, torch tensor format
    :param treeG: a tree with a Haar Global Orthonormal Basis
    :param num_high_pass: number of high passes {1, 2, 3}
    :return: high-pass and low-pass coefficients (in time domain) arranged in the following format:
                e.g., [[*, *], [*, *], [*, *], *]
                This is an example of output with a 4-layer tree treeG and num_high_pass = 2.
                3 x [*, *] represent 3 groups of high-pass coefficients sets.
                1 x * represents 1 low-pass coefficients set.
                All *s are in torch tensor format.
    """
    J = len(treeG)  # level of decomposition
    # coefs = list([None] * J)
    fh = DFT_gL(f, treeG)

    for j in range(J - 1):
        nj = len(treeG[j]['clusters'])  # number of vertices in level j
        nj1 = len(treeG[j + 1]['clusters'])  # number of vertices in level j + 1
        # generate filter bank for level j
        ell = np.array([i for i in range(1, nj + 1)])
        if num_high_pass == 3:
            ha, hb1, hb2, hb3 = filter_bank_3high(ell, nj, nj1)
            ha = torch.tensor(ha).to(device)
            hb1 = torch.tensor(hb1).to(device)
            hb2 = torch.tensor(hb2).to(device)
            hb3 = torch.tensor(hb3).to(device)
            if j == 0:
                coefs = torch.mul(fh, hb1)  # fh is d x N
                if fh.shape[1] < max_tree_nodes[j]:
                    coefs = torch.cat((coefs, torch.zeros(coefs.shape[0], max_tree_nodes[j] -
                                                          fh.shape[1]).to(device)), 1)
            else:
                coefs = torch.cat((coefs, torch.mul(fh, hb1)), 1)
                if fh.shape[1] < max_tree_nodes[j]:
                    coefs = torch.cat((coefs, torch.zeros(coefs.shape[0], max_tree_nodes[j] -
                                                          fh.shape[1]).to(device)), 1)
            coefs = torch.cat((coefs, torch.mul(fh, hb2)), 1)
            if fh.shape[1] < max_tree_nodes[j]:
                coefs = torch.cat((coefs, torch.zeros(coefs.shape[0], max_tree_nodes[j] -
                                                      fh.shape[1]).to(device)), 1)
            coefs = torch.cat((coefs, torch.mul(fh, hb3)), 1)
            if fh.shape[1] < max_tree_nodes[j]:
                coefs = torch.cat((coefs, torch.zeros(coefs.shape[0], max_tree_nodes[j] -
                                                      fh.shape[1]).to(device)), 1)
        elif num_high_pass == 2:
            ha, hb1, hb2 = filter_bank_2high(ell, nj, nj1)
            ha = torch.tensor(ha).to(device)
            hb1 = torch.tensor(hb1).to(device)
            hb2 = torch.tensor(hb2).to(device)
            if j == 0:
                coefs = torch.mul(fh, hb1)  # fh is d x N
                if fh.shape[1] < max_tree_nodes[j]:
                    coefs = torch.cat((coefs, torch.zeros(coefs.shape[0], max_tree_nodes[j] -
                                                          fh.shape[1]).to(device)), 1)
            else:
                coefs = torch.cat((coefs, torch.mul(fh, hb1)), 1)
                if fh.shape[1] < max_tree_nodes[j]:
                    coefs = torch.cat((coefs, torch.zeros(coefs.shape[0], max_tree_nodes[j] -
                                                          fh.shape[1]).to(device)), 1)
            coefs = torch.cat((coefs, torch.mul(fh, hb2)), 1)
            if fh.shape[1] < max_tree_nodes[j]:
                coefs = torch.cat((coefs, torch.zeros(coefs.shape[0], max_tree_nodes[j] -
                                                      fh.shape[1]).to(device)), 1)
        elif num_high_pass == 1:
            ha, hb1 = filter_bank_1high(ell, nj, nj1)
            ha = torch.tensor(ha).to(device)
            hb1 = torch.tensor(hb1).to(device)
            if j == 0:
                coefs = torch.mul(fh, hb1)  # fh is d x N
                if fh.shape[1] < max_tree_nodes[j]:
                    coefs = torch.cat((coefs, torch.zeros(coefs.shape[0], max_tree_nodes[j] -
                                                          fh.shape[1]).to(device)), 1)
            else:
                coefs = torch.cat((coefs, torch.mul(fh, hb1)), 1)
                if fh.shape[1] < max_tree_nodes[j]:
                    coefs = torch.cat((coefs, torch.zeros(coefs.shape[0], max_tree_nodes[j] -
                                                          fh.shape[1]).to(device)), 1)
        else:
            raise Exception('Invalid num_high_pass')
        # for low-pass filter
        fha = torch.mul(fh, ha)  # convolute with ha
        fh = dwn_f_hat(fha, nj1)  # downsampling, fh is a torch tensor

    # coefs[-1] = fh
    coefs = torch.cat((coefs, fh), 1)
    if fh.shape[1] < max_tree_nodes[-1]:
        coefs = torch.cat((coefs, torch.zeros(coefs.shape[0], max_tree_nodes[-1] -
                                              fh.shape[1]).to(device)), 1)
    return coefs


def FGT_Reconstr_tensor(coefs, treeG, max_tree_nodes, num_high_pass=3):
    """
    This function performs FGT reconstruction which reconstructs the graph signal (in time domain)
    from a set of high-pass and low-pass coefficients (in time domain).
    [PyTorch-enabled Function]

    :param max_tree_nodes: max of nodes.
    :param num_high_pass: number of high passes {1, 2, 3}.
    :param coefs: a set of high-pass and low-pass coefficients (in time domain) in the format
                    [[*, *], [*, *], [*, *], ..., *] with each * in torch tensor format.
    :param treeG: a tree with a Haar Global Orthonormal Basis
    :return: a vector of graph signal (in time domain) in torch tensor format.
    """
    J = len(treeG)  # level of decomposition
    # compute fh from low-pass coefficients at level J
    # fh1 = coefs[-1].to(device)
    idx = len(treeG[-1]['clusters'])
    fh1 = coefs[:, -max_tree_nodes[-1]:]
    fh1 = fh1[:, :idx]
    cx = coefs[:, :-max_tree_nodes[-1] or None]
    # num_high_pass = len(coefs[0])

    for j in np.arange(0, J - 1)[::-1]:
        nj1 = len(treeG[j + 1]['clusters'])
        # n_{j - 1}, number of vertices in level j - 1
        nj = len(treeG[j]['clusters'])
        # n_{j}, number of vertices in level j
        # eigenvalues at levels j - 1 and j
        ell = np.array([i for i in range(1, nj + 1)])
        if num_high_pass == 3:
            # generate filter bank for level j
            ha, hb1, hb2, hb3 = filter_bank_3high(ell, nj, nj1)
            ha = torch.tensor(ha).to(device)
            hb1 = torch.tensor(hb1).to(device)
            hb2 = torch.tensor(hb2).to(device)
            hb3 = torch.tensor(hb3).to(device)
        elif num_high_pass == 2:
            # generate filter bank for level j
            ha, hb1, hb2 = filter_bank_2high(ell, nj, nj1)
            ha = torch.tensor(ha).to(device)
            hb1 = torch.tensor(hb1).to(device)
            hb2 = torch.tensor(hb2).to(device)
        elif num_high_pass == 1:
            # generate filter bank for level j
            ha, hb1 = filter_bank_1high(ell, nj, nj1)
            ha = torch.tensor(ha).to(device)
            hb1 = torch.tensor(hb1).to(device)
        else:
            raise Exception('Invalid num_high_pass')
        # for low-pass filter
        # upsampling and convolution of low-pass filter
        fh1 = up_f_hat(fh1, nj).to(device)  # upsampling for low-pass
        fh1 = torch.mul(fh1, ha)  # convolute with ha
        if num_high_pass == 3:
            idx_1 = len(treeG[j]['clusters'])

            fhb3 = cx[:, -max_tree_nodes[j]:]
            fhb3 = fhb3[:, :idx_1]
            cx = cx[:, :-max_tree_nodes[j] or None]

            fhb2 = cx[:, -max_tree_nodes[j]:]
            fhb2 = fhb2[:, :idx_1]
            cx = cx[:, :-max_tree_nodes[j] or None]

            fhb1 = cx[:, -max_tree_nodes[j]:]
            fhb1 = fhb1[:, :idx_1]
            cx = cx[:, :-max_tree_nodes[j] or None]

            fh1 += torch.mul(fhb1, hb1)
            fh1 += torch.mul(fhb2, hb2)
            fh1 += torch.mul(fhb3, hb3)
        elif num_high_pass == 2:
            idx_1 = len(treeG[j]['clusters'])

            fhb2 = cx[:, -max_tree_nodes[j]:]
            fhb2 = fhb2[:, :idx_1]
            cx = cx[:, :-max_tree_nodes[j] or None]

            fhb1 = cx[:, -max_tree_nodes[j]:]
            fhb1 = fhb1[:, :idx_1]
            cx = cx[:, :-max_tree_nodes[j] or None]

            fh1 += torch.mul(fhb1, hb1)
            fh1 += torch.mul(fhb2, hb2)
        elif num_high_pass == 1:
            idx_1 = len(treeG[j]['clusters'])

            fhb1 = cx[:, -max_tree_nodes[j]:]
            fhb1 = fhb1[:, :idx_1]
            cx = cx[:, :-max_tree_nodes[j] or None]

            fh1 += torch.mul(fhb1, hb1)
        else:
            raise Exception('Invalid num_high_pass')
    # aDFT to evaluate f
    f_rec = aDFT_gL(fh1, treeG)

    return f_rec


class FGConv(nn.Module):
    def __init__(self, in_features, out_features, num_coefs, bias=True):
        super(FGConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_coefs = num_coefs
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features).cuda())
            self.filter = nn.Parameter(torch.Tensor(num_coefs).cuda())
        else:
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
            self.filter = nn.Parameter(torch.Tensor(num_coefs))
        if bias:
            if torch.cuda.is_available():
                self.bias = nn.Parameter(torch.Tensor(out_features).cuda())
            else:
                self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.filter, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, treeG_batch, batch, num_high_pass, max_tree_nodes):
        batch_size = int(batch.max() + 1)
        x = torch.matmul(x, self.weight)

        for i in range(batch_size):
            bi = (batch == i)
            if i == 0:
                coefs = FGT_Decomp_tensor(x[bi, :].T, treeG_batch[i], max_tree_nodes, num_high_pass)
                batch_re = torch.ones(self.out_features, dtype=torch.long) * i
            else:
                coefs = torch.cat((coefs,
                                   FGT_Decomp_tensor(x[bi, :].T, treeG_batch[i], max_tree_nodes, num_high_pass)), 0)
                batch_re = torch.cat((batch_re, torch.ones(self.out_features, dtype=torch.long) * i))

        cx = torch.mul(coefs, self.filter)

        for i in range(batch_size):
            bi = (batch_re == i)
            if i == 0:
                x_out = FGT_Reconstr_tensor(cx[bi, :], treeG_batch[i], max_tree_nodes, num_high_pass).T
            else:
                x_out = torch.cat((x_out,
                                   FGT_Reconstr_tensor(cx[bi, :], treeG_batch[i], max_tree_nodes, num_high_pass).T), 0)

        if self.bias is not None:
            x_out = x_out + self.bias
        return x_out


class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, num_coefs, num_high_pass, max_tree_nodes):
        super(Net, self).__init__()
        self.num_high_pass = num_high_pass
        self.max_tree_nodes = max_tree_nodes

        self.conv1 = FGConv(num_features, nhid, num_coefs)
        self.conv2 = FGConv(nhid, nhid, num_coefs)

        self.lin1 = nn.Linear(nhid, nhid // 2)
        self.m1 = nn.BatchNorm1d(nhid // 2)
        self.lin2 = nn.Linear(nhid // 2, num_classes)

    def forward(self, data):
        x, batch, tree = data.x.type(torch.DoubleTensor).to(device), data.batch, data.tree

        x = F.relu(self.conv1(x, tree, batch, self.num_high_pass, self.max_tree_nodes))
        x = F.relu(self.conv2(x, tree, batch, self.num_high_pass, self.max_tree_nodes))

        x = scatter_add(x, batch, dim=0)

        x = F.relu(self.m1(self.lin1(x)))
        x = F.log_softmax(self.lin2(x), dim=-1)

        return x


def edge_to_adj(edge_index, num_nodes, edge_weight=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ))
    adj = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([num_nodes, num_nodes]))
    return adj


def MyDataset(dataset, num_tree_layers):
    dataset1 = list()
    max_tree_nodes = [0] * num_tree_layers
    for i in range(len(dataset)):
        data1 = Data(x=dataset[i].x, y=dataset[i].y)
        treeG = SC_tree(edge_to_adj(dataset[i].edge_index, dataset[i].num_nodes).to_dense().numpy(), num_tree_layers)
        treeG = addweight1(treeG)
        tree_Haar = HaarGOB(treeG)
        data1.tree = tree_Haar
        max_tree_nodes = list(map(max, [len(treeG[k]['clusters']) for k in range(num_tree_layers)], max_tree_nodes))
        dataset1.append(data1)
    return dataset1, max_tree_nodes


def test(model, loader, device):
    model.eval()
    correct = 0.
    loss = 0.  # edited by Ming with concern for further extension
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out, data.y, reduction='sum').item()
    return correct / len(loader.dataset), loss / len(loader.dataset)


dataname = 'DD'
path = osp.join(os.path.abspath(''), 'data', dataname)
dataset = TUDataset(path, name=dataname)

num_features = dataset.num_features
num_classes = dataset.num_classes

dataset, max_tree_nodes = MyDataset(dataset, num_tree_layers=2)

num_training = int(len(dataset) * 0.8)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)

# Parameter Setting
batch_size = 64
learning_rate = 0.0005
weight_decay = 1e-3
nhid = 128
epochs = 30
early_stopping = 200
num_reps = 2
num_high_pass = 3

# Number of FGT coefficients
num_coefs = max_tree_nodes[-1]
J = len(max_tree_nodes)
for j in np.arange(0, J - 1)[::-1]:
    num_coefs = num_coefs + num_high_pass * max_tree_nodes[j]
print('Number of FGT coefficients: {}'.format(num_coefs))

# create results matrix
epoch_train_loss = np.zeros((num_reps, epochs))
epoch_train_acc = np.zeros((num_reps, epochs))
epoch_valid_loss = np.zeros((num_reps, epochs))
epoch_valid_acc = np.zeros((num_reps, epochs))
epoch_test_loss = np.zeros((num_reps, epochs))
epoch_test_acc = np.zeros((num_reps, epochs))
saved_model_loss = np.zeros(num_reps)
saved_model_acc = np.zeros(num_reps)

for r in range(num_reps):
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=False)
    # add validation for a possible early stopping
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = Net(num_features, nhid, num_classes, num_coefs, num_high_pass, max_tree_nodes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training
    min_loss = 1e10
    patience = 0
    print("****** Training start ******")
    for epoch in range(epochs):
        print("* Epoch: {:03d} ...".format(epoch + 1))
        model.train()
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
        train_acc, train_loss = test(model, train_loader, device)
        print("  Training loss: {:5f}, accuracy: {:.5f}".format(train_loss, train_acc))
        val_acc, val_loss = test(model, val_loader, device)
        print("  Validation loss: {:5f}, accuracy: {:.5f}".format(val_loss, val_acc))
        test_acc, test_loss = test(model, test_loader, device)
        print("  Test loss: {:5f}, accuracy: {:.5f}".format(test_loss, test_acc))
        if val_loss < min_loss:
            torch.save(model.state_dict(), 'latest.pth')  # save the model and reuse later in test
            print("= Model saved at epoch: {:03d}".format(epoch + 1))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > early_stopping:
            break

        epoch_train_loss[r, epoch] = train_loss
        epoch_train_acc[r, epoch] = train_acc
        epoch_valid_loss[r, epoch] = val_loss
        epoch_valid_acc[r, epoch] = val_acc
        epoch_test_loss[r, epoch] = test_loss
        epoch_test_acc[r, epoch] = test_acc

    # Test
    print("****** Test start ******")
    model = Net(num_features, nhid, num_classes, num_coefs, num_high_pass, max_tree_nodes).to(device)
    model.load_state_dict(torch.load('latest.pth'))
    test_acc, test_loss = test(model, test_loader, device)
    print("Test accuracy: {:.5f}".format(test_acc))

    saved_model_loss[r] = test_loss
    saved_model_acc[r] = test_acc

# save the results
np.savez("results.npz",
         epoch_train_loss=epoch_train_loss,
         epoch_train_acc=epoch_train_acc,
         epoch_valid_loss=epoch_valid_loss,
         epoch_valid_acc=epoch_valid_acc,
         epoch_test_loss=epoch_test_loss,
         epoch_test_acc=epoch_test_acc,
         saved_model_loss=saved_model_loss,
         saved_model_acc=saved_model_acc)
