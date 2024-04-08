import numpy as np
import pprint
import networkx as nx
import scipy as sp
from mimiciii_teg.schemas.schemas import DECIMALS

import copy



def IPC_nx(G, x):
    #C = dict.fromkeys(G, 0.0)
    x = x.astype(np.float64)
    n = G.number_of_nodes()
    C = np.zeros(n)
    w_s_r = np.float64(0)
    w_v = np.zeros(n, dtype=np.float64)
    # compute relative contribution coefficients (weights)
    for s in range(n):
        for r in range(n):
            if x[r] - x[s] > 0:
                w_s_r += x[r] - x[s]
                w_v[s] += x[r] - x[s]
                w_v[r] += x[r] - x[s]
    D = dict(nx.all_pairs_dijkstra(G, weight='weight'))
    paths = dict()
    for v in range(n):
        if v not in D:
            continue
        for s in D:
            if v not in D[s][0]:
                continue
            for r in D[s][0]:
                if r not in D[v][0] or x[r] - x[s] <= 0:
                    continue
                elif s == v or r == v or s == r:
                    continue
                elif np.round(D[s][0][r], DECIMALS) == np.round(D[s][0][v] + D[v][0][r], DECIMALS):
                    if (s, r) not in paths:
                        tmp_paths = nx.all_shortest_paths(G, source=s, target=r, weight='weight')
                        paths[(s, r)] = [list(p) for p in set(tuple(p) for p in tmp_paths)]
                    sigma_sr = np.float64(len(paths[(s, r)]))
                    sv_paths = list(nx.all_shortest_paths(G, source=s, target=v, weight='weight'))
                    #sv_paths = [list(p) for p in set(tuple(p) for p in tmp_paths)]
                    vr_paths = list(nx.all_shortest_paths(G, source=v, target=r, weight='weight'))
                    #vr_paths = [list(p) for p in set(tuple(p) for p in tmp_paths)]
                    sigma_v_sr = np.float64(len(sv_paths) * len(vr_paths))
                    w_v_sr = (x[r] - x[s]) / (w_s_r - w_v[v]) 
                    C[v] += sigma_v_sr / sigma_sr * w_v_sr
                    '''
                    # debugging NetworX using the network in experiments/example.py
                    # NetworkX has some numerical instability and doesn't satisfy Bellman criterion
                    # of sigma_sr == sigma_sv * sigma_vr
                    if s == 1 and r == 23 and v == 4:
                        print('C_nx', s, r, v, len(sv_paths), len(vr_paths), sigma_sr, w_v_sr, C[v])
                        print('C_nx sv_paths', sv_paths)
                        print('C_nx vr_paths', vr_paths)
                        print('C_nx sr_paths', paths[(s, r)])
                        sr_weights = []
                        for p in paths[(s, r)]:
                            weight = 0
                            v1 = p[0]
                            for v2 in p[1:]:
                                weight += G[v1][v2]["weight"]
                                v1 = v2
                            print('sr path', p, weight)
                            sr_weights.append(weight)
                        sr_w = weight
                        sv_weights = []
                        sr_weights_TF = []
                        for p in sv_paths:
                            weight = 0
                            v1 = p[0]
                            for v2 in p[1:]:
                                weight += G[v1][v2]["weight"]
                                v1 = v2
                            print('sv path', p, weight)
                            sv_weights.append(weight)
                        print('sv_weights', sv_weights)
                        vr_weights = []
                        for p in vr_paths:
                            weight = 0
                            v1 = p[0]
                            for v2 in p[1:]:
                                weight += G[v1][v2]["weight"]
                                v1 = v2
                            print('vr path', p, weight)
                            sr_weights.append([val + weight for val in sv_weights])
                            sr_weights_TF.append([val + weight == sr_w for val in sv_weights])
                            vr_weights.append(weight)
                        print('vr_weights', vr_weights)
                        pprint.pprint(sr_weights)
                        pprint.pprint(sr_weights_TF)
                    '''
    return C


def IPC_with_paths_nx(G, x):
    '''
    Inverse percolation centrality algorithm using Networkx
    '''
    #C = dict.fromkeys(G, 0.0)
    n = G.number_of_nodes()
    C = np.zeros(n, dtype=np.float64)
    w_s_r = np.float64(0)
    w_v = np.zeros(n, dtype=np.float64)
    # compute relative contribution coefficients (weights)
    for s in range(n):
        for r in range(n):
            if x[r] - x[s] > 0:
                w_s_r += x[r] - x[s]
                w_v[s] += x[r] - x[s]
                w_v[r] += x[r] - x[s]
    D = dict(nx.all_pairs_dijkstra(G, weight='weight'))
    v_paths = dict()
    paths = dict()
    for v in range(n):
        if v not in D:
            continue
        for s in D:
            if v not in D[s][0]:
                continue
            for r in D[s][0]:
                if r not in D[v][0] or x[r] - x[s] <= 0:
                    continue
                elif s == v or r == v or s == r:
                    continue
                elif np.round(D[s][0][r], DECIMALS) == np.round(D[s][0][v] + D[v][0][r], DECIMALS):
                    if (s, r) not in paths:
                        tmp_paths = nx.all_shortest_paths(G, source=s, target=r, weight='weight')
                        paths[(s, r)] = [list(p) for p in set(tuple(p) for p in tmp_paths)]
                    sigma_sr = len(paths[(s, r)])
                    # making sure that the paths are unique
                    tmp_paths = nx.all_shortest_paths(G, source=s, target=v, weight='weight')
                    sv_paths = [list(p) for p in set(tuple(p) for p in tmp_paths)]
                    tmp_paths = nx.all_shortest_paths(G, source=v, target=r, weight='weight')
                    vr_paths = [list(p) for p in set(tuple(p) for p in tmp_paths)]
                    sigma_v_sr = np.float64(len(sv_paths) * len(vr_paths))
                    w_v_sr = (x[r] - x[s]) / (w_s_r - w_v[v]) 
                    C[v] += sigma_v_sr / sigma_sr * w_v_sr
                    if v not in v_paths:
                        v_paths[v] = []
                    #for p in paths[(s, r)]:
                    #    if v in p:
                    #        v_paths[v].append(p)
                    for p1 in sv_paths:
                        for p2 in vr_paths:
                            v_paths[v].append(p1 + p2[1:])
    return C, v_paths, paths


def IPC_nx_v1(G, states):
    states = states.astype(np.float64)
    C = dict.fromkeys(G, 0.0)
    n = G.number_of_nodes()
    w_s_r = 0
    w_v = np.zeros(n, dtype=np.float64)
    targets = dict()
    # compute relative contribution coefficients (weights)
    for s in range(n):
        for r in range(n):
            if x[r] - x[s] > 0:
                w_s_r += x[r] - x[s]
                w_v[s] += x[r] - x[s]
                w_v[r] += x[r] - x[s]
                if s not in targets:
                    targets[s] = []
                targets[s].append(r)
    paths = dict()
    v_paths = dict()
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
            paths[(s, t)] = st_paths
    for v in v_set:
        for s in targets:
            for t in targets[s]:
                if (s, v) in D and (v, t) in D:
                    sv_len = D[s, v][0]
                    sv_paths = D[s, v][1]
                    vt_len = D[v, t][0]
                    vt_paths = D[v, t][1]
                    if s != v and t != v and s != t and D[s, t][0] == sv_len + vt_len:
                        w_v_sr = (x[r] - x[s]) / (w_s_r - w_v[v])
                        sigma_v_st = np.float64(len(sv_paths) * len(vt_paths))
                        sigma_st = np.float64(len(D[s, t][1]))
                        if sigma_st == 0:
                            continue
                        C[v] += sigma_v_st / sigma_st * w_v_sr
                        if v not in paths:
                            paths[v] = []
                        for p1 in sv_paths:
                            for p2 in vt_paths:
                                paths[v].append(p1 + p2[1:])
    return C, v_paths, paths
