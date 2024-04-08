from math import isclose
import numpy as np
import pprint
import networkx as nx
import scipy as sp
from mimiciii_teg.schemas.schemas import DECIMALS

import copy


def IPC_dense(A, x):
    '''
    Inverse centrality algorithm
    '''
    x = x.astype(np.float64)
    # adjacency matrix
    n = A.shape[0]
    # centrality
    C = np.zeros(n, dtype=np.float64)
    # shortest path distance matrix
    D_d = np.zeros((n, n), dtype=np.float64)
    D_d[:, :] = np.float64('inf')
    # shortest path count matrix
    D_s = np.zeros((n, n), dtype=np.float64)
    # initiate D_d and D_s
    for i in range(n):
        D_d[i, i] = 0
        D_s[i, i] = 0
    for k, v in A.items():
        i = k[0]
        j = k[1]
        if i != j and v != 0: 
            D_d[i, j] = v
            D_s[i, j] = 1
    w_s_r = 0
    w_v = np.zeros(n, dtype=np.float64)
    # compute relative contribution coefficients (weights)
    for s in range(n):
        for r in range(n):
            if x[r] - x[s] > 0:
                w_s_r += x[r] - x[s]
                w_v[s] += x[r] - x[s]
                w_v[r] += x[r] - x[s]
    # iterate for each vertex
    for k in range(n):
        # iterate for each vertex
        for i in range(n):
            # iterate for each vertex
            for j in range(n):
                # shortest paths through k exists
                if D_d[i, k] == np.float64('inf') or D_d[k, j] == np.float64('inf'):
                    continue
                if np.round(D_d[i, k] + D_d[k, j], DECIMALS) < np.round(D_d[i, j], DECIMALS):
                    D_d[i, j] = D_d[i, k] + D_d[k, j]
                    D_s[i, j] = D_s[i, k] * D_s[k, j]
                # other shortest paths through k exist
                elif np.round(D_d[i, k] + D_d[k, j], DECIMALS) == np.round(D_d[i, j], DECIMALS):
                    D_s[i, j] += D_s[i, k] * D_s[k, j]
    # iterate for each vertex
    for v in range(n):
        # iterate for each vertex
        for s in range(n):
            # iterate for each vertex
            for r in range(n):
                if x[r] - x[s] <= 0 or D_s[s, r] == 0:
                    continue
                if D_d[s, v] == np.float64('inf') or D_d[v, r] == np.float64('inf'):
                    continue
                elif s == v or r == v or s == r:
                    continue
                elif np.round(D_d[s, r], DECIMALS) == np.round(D_d[s, v] + D_d[v, r], DECIMALS):
                    # the weight
                    w_v_sr = (x[r] - x[s]) / (w_s_r - w_v[v])
                    # accummulate centrality value
                    C[v] += (D_s[s, v] * D_s[v, r]) / D_s[s, r] * w_v_sr
    return C


def IPC_sparse(A, x):
    '''
    Inverse centrality algorithm
    '''
    x = x.astype(np.float64)
    n = A.shape[0]
    # centrality
    C = np.zeros(n, dtype=np.float64)
    # shortest path distance matrix
    D_d = copy.deepcopy(A)
    # shortest path count matrix
    D_s = sp.sparse.dok_matrix((n, n), dtype=np.float64)
    for i in range(n):
        D_d[i, i] = 0
        D_s[i, i] = 1
    for k, v in A.items():
        i = k[0]
        j = k[1]
        if i != j and v != 0: 
            D_s[i, j] = 1
    w_s_r = 0
    w_v = np.zeros(n, dtype=np.float64)
    # compute relative contribution coefficients (weights)
    for s in range(n):
        for r in range(n):
            if x[r] - x[s] > 0:
                w_s_r += x[r] - x[s]
                w_v[s] += x[r] - x[s]
                w_v[r] += x[r] - x[s]
    # iterate for each vertex
    for k in range(n):
        # if k-th col is empty, continue
        if D_d.getcol(k).count_nonzero() == 0:
            continue
        # iterate over nonzero column indices
        for i in D_d.getcol(k).nonzero()[0]:
            if D_d.getrow(k).count_nonzero() == 0:
                continue
            # iterate over nonzero row indices
            for j in D_d.getrow(k).nonzero()[1]:
                # check the entry for i, j
                if not D_d.get((i, j), False):
                    d_ij = np.float64('inf')
                else:
                    d_ij = D_d[i, j]
                # shortest paths through k exists
                if np.round(D_d[i, k] + D_d[k, j], DECIMALS) < np.round(d_ij, DECIMALS):
                    D_d[i, j] = D_d[i, k] + D_d[k, j]
                    D_s[i, j] = D_s[i, k] * D_s[k, j]
                # other shortest paths through k exist
                elif np.round(D_d[i, k] + D_d[k, j], DECIMALS) == np.round(D_d[i, j], DECIMALS):
                    D_s[i, j] += D_s[i, k] * D_s[k, j]
    for v in range(n):
        # if v-th col is empty, continue
        if D_d.getcol(v).count_nonzero() == 0:
            continue
        # iterate over nonzero column indices
        for s in D_d.getcol(v).nonzero()[0]:
            # if v-th row is empty, continue
            if D_d.getrow(v).count_nonzero() == 0:
                continue
            # iterate over nonzero row indices
            for r in D_d.getrow(v).nonzero()[1]:
                if s == r or s == v or r == v or x[r] - x[s] <= 0:
                    continue
                elif not D_d.get((s, v), False) or not D_d.get((v, r), False):
                    continue
                elif np.round(D_d[s, r], DECIMALS) == np.round(D_d[s, v] + D_d[v, r], DECIMALS):
                    # the weight
                    w_v_sr = (x[r] - x[s]) / (w_s_r - w_v[v])
                    # accummulate centrality value
                    C[v] += (D_s[s, v] * D_s[v, r]) / D_s[s, r] * w_v_sr
    return C
