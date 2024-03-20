import numpy as np
import pprint
import networkx as nx
import scipy as sp

import copy


def IPC_with_target_nx(G, x):
    #IPC = dict.fromkeys(G, 0.0)
    n = G.number_of_nodes()
    IPC = np.zeros(n)
    w_s_r = 0
    w_v_r = np.zeros(n)
    # compute relative contribution coefficients (weights)
    for s in range(n):
        for r in range(n):
            if x[r] - x[s] > 0:
                w_s_r += x[r] - x[s]
                w_v_r[s] += x[r] - x[s]
    D = dict(nx.all_pairs_dijkstra(G, weight='weight'))
    paths = dict()
    for v in range(n):
        if v not in D:
            continue
        for s in D:
            if v not in D[s][0]:
                continue
            for r in D[s][0]:
                if r not in D[v][0]:
                    continue
                if x[r] - x[s] <= 0:
                    continue
                if s != v and r != v and s != r and D[s][0][r] == D[s][0][v] + D[v][0][r]:
                    if (s, r) not in paths:
                        paths[(s, r)] = list(nx.all_shortest_paths(G, source=s, target=r, weight='weight'))
                    sigma_sr = len(paths[(s, r)])
                    if sigma_sr == 0:
                        continue
                    sv_paths = list(nx.all_shortest_paths(G, source=s, target=v, weight='weight'))
                    vr_paths = list(nx.all_shortest_paths(G, source=v, target=r, weight='weight'))
                    sigma_v_sr = float(len(sv_paths) * len(vr_paths))
                    w_v_sr = (x[r] - x[s]) / (w_s_r - w_v_r[v]) 
                    IPC[v] += sigma_v_sr / sigma_sr * w_v_sr
    return IPC

def IPC_with_target_path_nx(G, x):
    '''
    Inverse percolation centrality algorithm using Networkx
    '''
    #IPC = dict.fromkeys(G, 0.0)
    n = G.number_of_nodes()
    IPC = np.zeros(n)
    w_s_r = 0
    w_v_r = np.zeros(n)
    # compute relative contribution coefficients (weights)
    for s in range(n):
        for r in range(n):
            if x[r] - x[s] > 0:
                w_s_r += x[r] - x[s]
                w_v_r[s] += x[r] - x[s]
    D = dict(nx.all_pairs_dijkstra(G, weight='weight'))
    v_paths = dict()
    paths = dict()
    V = set()
    for v in range(n):
        if v not in D:
            continue
        for s in D:
            if v not in D[s][0]:
                continue
            for r in D[s][0]:
                if r not in D[v][0]:
                    continue
                if x[r] - x[s] <= 0:
                    continue
                if s != v and r != v and s != r and D[s][0][r] == D[s][0][v] + D[v][0][r]:
                    if (s, r) not in paths:
                        paths[(s, r)] = list(nx.all_shortest_paths(G, source=s, target=r, weight='weight'))
                    sigma_sr = len(paths[(s, r)])
                    if sigma_sr == 0:
                        continue
                    sv_paths = list(nx.all_shortest_paths(G, source=s, target=v, weight='weight'))
                    vr_paths = list(nx.all_shortest_paths(G, source=v, target=r, weight='weight'))
                    sigma_v_sr = float(len(sv_paths) * len(vr_paths))
                    w_v_sr = (x[r] - x[s]) / (w_s_r - w_v_r[v]) 
                    if v not in v_paths:
                        v_paths[v] = []
                    IPC[v] += sigma_v_sr / sigma_sr * w_v_sr
                    for p in paths[(s, r)]:
                        if v in p:
                            v_paths[v].append(p)
                            V.update(p)
    return IPC, V, v_paths, paths


def IPC_dense(G, x):
    '''
    Inverse centrality algorithm
    '''
    # adjacency matrix
    A = nx.adjacency_matrix(G)
    A = A.todok(copy=True)
    n = A.shape[0]
    # centrality
    C = np.zeros(n)
    # shortest path distance matrix
    D_d = np.zeros((n, n), dtype=float)
    D_d[:, :] = float('inf')
    # shortest path count matrix
    D_s = np.zeros((n, n), dtype=float)
    # initiate D_d and D_s
    for i in range(n):
        D_d[i, i] = 0
        D_s[i, i] = 1
    for k, v in A.items():
        i = k[0]
        j = k[1]
        if i != j and v != 0: 
            D_d[i, j] = v
            D_s[i, j] = 1
    w_s_r = 0
    w_v_r = np.zeros(n)
    # compute relative contribution coefficients (weights)
    for s in range(n):
        for r in range(n):
            if x[r] - x[s] > 0:
                w_s_r += x[r] - x[s]
                w_v_r[s] += x[r] - x[s]
    # iterate for each vertex
    for k in range(n):
        # iterate for each vertex
        for i in range(n):
            # iterate for each vertex
            for j in range(n):
                # shortest paths through k exists
                if D_d[i, k] + D_d[k, j] < D_d[i, j]:
                    D_d[i, j] = D_d[i, k] + D_d[k, j]
                    D_s[i, j] = D_s[i, k] * D_s[k, j]
                # other shortest paths through k exist
                elif D_d[i, k] + D_d[k, j] == D_d[i, j] and i != j:
                    D_s[i, j] += D_s[i, k] * D_s[k, j]
    # iterate for each vertex
    for v in range(n):
        # iterate for each vertex
        for s in range(n):
            # iterate for each vertex
            for r in range(n):
                if s != v and r != v and D_d[s, r] == D_d[s, v] + D_d[v, r] and x[r] - x[s] > 0:
                    # the weight
                    w_v_sr = (x[r] - x[s]) / (w_s_r - w_v_r[v])
                    # accummulate centrality value
                    C[v] += (D_s[s, v] * D_s[v, r]) / D_s[s, r] * w_v_sr
    return C


def IPC_sparse(G, x):
    '''
    Inverse centrality algorithm
    '''
    # adjacency matrix
    A = nx.adjacency_matrix(G)
    A = A.todok(copy=True)
    n = A.shape[0]
    # centrality
    C = np.zeros(n)
    # shortest path distance matrix
    D_d = copy.deepcopy(A)
    # shortest path count matrix
    D_s = sp.sparse.dok_matrix((n, n), dtype=float)
    for i in range(n):
        D_s[i, i] = 1
        D_d[i, i] = 0
    for k, v in A.items():
        i = k[0]
        j = k[1]
        if i != j and v != 0: 
            D_s[i, j] = 1
    w_s_r = 0
    w_v_r = np.zeros(n)
    # compute relative contribution coefficients (weights)
    for s in range(n):
        for r in range(n):
            if x[r] - x[s] > 0:
                w_s_r += x[r] - x[s]
                w_v_r[s] += x[r] - x[s]
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
                    d_ij = float('inf')
                else:
                    d_ij = D_d[i, j]
                # shortest paths through k exists
                if D_d[i, k] + D_d[k, j] < d_ij:
                    D_d[i, j] = D_d[i, k] + D_d[k, j]
                    D_s[i, j] = D_s[i, k] * D_s[k, j]
                # other shortest paths through k exist
                elif D_d[i, k] + D_d[k, j] == d_ij and i != j:
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
                if not D_d.get((s, v), False) or not D_d.get((v, r), False):
                    continue
                if D_d[s, r] == D_d[s, v] + D_d[v, r]:
                    # the weight
                    w_v_sr = (x[r] - x[s]) / (w_s_r - w_v_r[v])
                    # accummulate centrality value
                    C[v] += (D_s[s, v] * D_s[v, r]) / D_s[s, r] * w_v_sr
    return C

def IPC_with_target_path_nx_v0(G, states):
    '''
    Inverse percolation centrality algorithm using Networkx
    '''
    #IPC = dict.fromkeys(G, 0.0)
    n = G.number_of_nodes()
    IPC = np.zeros(n)
    S = 0.0
    for i in range(n):
        deltas = states - states[i]
        S += np.sum(deltas[deltas > 0.0])
    D = dict(nx.all_pairs_dijkstra(G, weight='weight'))
    v_paths = dict()
    paths = dict()
    V = set()
    for v in range(n):
        if v not in D:
            continue
        S_exclude_v = S
        deltas_v_source = states - states[v]
        S_exclude_v -= np.sum(deltas_v_source[deltas_v_source > 0])
        deltas_v_target = states * (-1) + states[v]
        S_exclude_v  -= np.sum(deltas_v_target[deltas_v_target > 0])
        for s in D:
            if v not in D[s][0]:
                continue
            for t in D[s][0]:
                if t not in D[v][0]:
                    continue
                delta = float(states[t] - states[s])
                if delta <= 0:
                    continue
                if s != v and t != v and s != t and D[s][0][t] == D[s][0][v] + D[v][0][t]:
                    if (s, t) not in paths:
                        paths[(s,t)] = list(nx.all_shortest_paths(G, source=s, target=t, weight='weight'))
                    sigma_st = len(paths[(s,t)])
                    if sigma_st == 0:
                        continue
                    sv_paths = list(nx.all_shortest_paths(G, source=s, target=v, weight='weight'))
                    vt_paths = list(nx.all_shortest_paths(G, source=v, target=t, weight='weight'))
                    sigma_v_st = float(len(sv_paths) * len(vt_paths))
                    w = delta/S_exclude_v 
                    if v not in v_paths:
                        v_paths[v] = []
                    IPC[v] += sigma_v_st / sigma_st * w
                    for p in paths[(s, t)]:
                        if v in p:
                            v_paths[v].append(p)
                            V.update(p)
    return IPC, V, v_paths, paths

def IPC_with_target_nx_v0(G, states):
    #IPC = dict.fromkeys(G, 0.0)
    n = G.number_of_nodes()
    IPC = np.zeros(n)
    S = 0.0
    for i in range(n):
        deltas = states - states[i]
        S += np.sum(deltas[deltas > 0.0])
    D = dict(nx.all_pairs_dijkstra(G, weight='weight'))
    paths = dict()
    for v in range(n):
        if v not in D:
            continue
        S_exclude_v = S
        deltas_v_source = states - states[v]
        S_exclude_v -= np.sum(deltas_v_source[deltas_v_source > 0])
        deltas_v_target = states * (-1) + states[v]
        S_exclude_v  -= np.sum(deltas_v_target[deltas_v_target > 0])
        for s in D:
            if v not in D[s][0]:
                continue
            for t in D[s][0]:
                if t not in D[v][0]:
                    continue
                delta = float(states[t] - states[s])
                if delta <= 0:
                    continue
                if s != v and t != v and s != t and D[s][0][t] == D[s][0][v] + D[v][0][t]:
                    if (s, t) not in paths:
                        paths[(s,t)] = list(nx.all_shortest_paths(G, source=s, target=t, weight='weight'))
                    sigma_st = len(paths[(s,t)])
                    if sigma_st == 0:
                        continue
                    sv_paths = list(nx.all_shortest_paths(G, source=s, target=v, weight='weight'))
                    vt_paths = list(nx.all_shortest_paths(G, source=v, target=t, weight='weight'))
                    sigma_v_st = float(len(sv_paths) * len(vt_paths))
                    w = delta/S_exclude_v 
                    IPC[v] += sigma_v_st / sigma_st * w
    return IPC


def IPC_with_target_path_nx_v1(G, x):
    '''
    Inverse percolation centrality algorithm using Networkx
    '''
    n = G.number_of_nodes()
    IPC = np.zeros(n)
    w_s_r = 0
    w_v_r = np.zeros(n)
    # compute relative contribution coefficients (weights)
    for s in range(n):
        for r in range(n):
            if x[r] - x[s] > 0:
                w_s_r += x[r] - x[s]
                w_v_r[s] += x[r] - x[s]
    D = dict(nx.all_pairs_dijkstra(G, weight='weight'))
    v_paths = dict()
    paths = dict()
    V = set()
    for v in range(n):
        if v not in D:
            continue
        for s in D:
            if v not in D[s][0]:
                continue
            for r in D[s][0]:
                if r not in D[v][0] or x[r] - x[s] <= 0:
                    continue
                if s != v and r != v and s != r and D[s][0][r] == D[s][0][v] + D[v][0][r]:
                    if (s, r) not in paths:
                        paths[(s, r)] = list(nx.all_shortest_paths(G, source=s, target=r, weight='weight'))
                    sigma_sr = len(paths[(s, r)])
                    if sigma_sr == 0:
                        continue
                    sv_paths = list(nx.all_shortest_paths(G, source=s, target=v, weight='weight'))
                    vr_paths = list(nx.all_shortest_paths(G, source=v, target=r, weight='weight'))
                    sigma_v_sr = float(len(sv_paths) * len(vr_paths))
                    w_v_sr = (x[r] - x[s]) / (w_s_r - w_v_r[v]) 
                    IPC[v] += sigma_v_sr / sigma_sr * w_v_sr
                    if v not in v_paths:
                        v_paths[v] = []
                    for p1 in sv_paths:
                        for p2 in vr_paths:
                            v_paths[v].append(p1 + p2[1:])
                            V.update(p1 + p2[1:])
    return IPC, V, v_paths, paths

def IPC_with_target_v2(G, states):
    IPC = dict.fromkeys(G, 0.0)
    n = G.number_of_nodes()
    w_s_r = 0
    w_v_r = np.zeros(n)
    # compute relative contribution coefficients (weights)
    for s in range(n):
        for r in range(n):
            if x[r] - x[s] > 0:
                w_s_r += x[r] - x[s]
                w_v_r[s] += x[r] - x[s]
    S = 0.0
    targets = dict()
    S_vt = dict()
    S_sv = dict()
    for v in range(n):
        deltas = states - states[v]
        S_vt[v] = np.sum(deltas[deltas > 0])
        if S_vt[v] != 0:
            targets[v] = [i for i, val in enumerate(deltas) if val > 0]
        S_vt[v] = np.sum(deltas[deltas > 0])
        deltas = states * (-1) + states[v]
        S_sv[v] = np.sum(deltas[deltas > 0])
        S += S_vt[v]
    paths = dict()
    D = dict()
    v_set = set()
    for s in targets:
        for t in targets[s]:
            try:
                length = nx.shortest_path_length(G, source=s, target=t, weight='weight')
                st_paths = list(nx.all_shortest_paths(G, source=s, target=t, weight='weight'))
            except nx.NetworkXNoPath:
                continue
            for path in st_paths:
                for v in path:
                    v_set.add(v)
            D[(s, t)] = [length, st_paths]
    for v in v_set:
        S_exclude_v = S - S_sv[v] - S_vt[v]
        if S_exclude_v == 0:
            continue
        for s in targets:
            for t in targets[s]:
                if (s, v) not in D:
                    try:
                        sv_len = nx.shortest_path_length(G, source=s, target=v, weight='weight')
                        sv_paths = list(nx.all_shortest_paths(G, source=s, target=v, weight='weight'))
                    except nx.NetworkXNoPath:
                        continue
                else:
                    sv_len = D[s, v][0]
                    sv_paths = D[s, v][1]
                if (v, t) not in D:
                    try:
                        vt_len = nx.shortest_path_length(G, source=v, target=t, weight='weight')
                        vt_paths = list(nx.all_shortest_paths(G, source=v, target=t, weight='weight'))
                    except nx.NetworkXNoPath:
                        continue
                else:
                    vt_len = D[v, t][0]
                    vt_paths = D[v, t][1]
                if s != v and t != v and s != t and D[s, t][0] == sv_len + vt_len:
                    w = (states[t] - states[s]) / S_exclude_v 
                    sigma_v_st = float(len(sv_paths) * len(vt_paths))
                    sigma_st = len(D[s, t][1])
                    if sigma_st == 0:
                        continue
                    if v not in paths:
                        paths[v] = []
                    IPC[v] += sigma_v_st / sigma_st * w
                    for p1 in sv_paths:
                        for p2 in vt_paths:
                            paths[v].append(p1 + p2[1:])
    return IPC, v_set, paths
