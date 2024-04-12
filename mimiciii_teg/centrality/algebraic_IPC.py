from pygraphblas import *
from pygraphblas import lib
from pygraphblas.types import Type, binop
import numpy as np
from pyvis import network as net
from scipy.sparse import dok_matrix
from mimiciii_teg.schemas.schemas import DECIMALS


class GS(Type):

    _base_name = "UDT"
    _numpy_t = None
    members = ['double w', 'int64_t n']
    one = (np.float64('inf'), np.int64(0))
    # equal operation is not working with "iseq" operation, D.iseq(D1)
    @binop(boolean=True)
    def EQ(z, x, y):
        if np.round(x.w, DECIMALS) == np.round(y.w, DECIMALS) and x.n == y.n:
            z = True
        else:
            z = False

    @binop()
    def PLUS(z, x, y):
        if np.round(x.w, DECIMALS) < np.round(y.w, DECIMALS):
            z.w = x.w
            z.n = x.n
        elif np.round(x.w, DECIMALS) == np.round(y.w, DECIMALS):
            z.w = x.w
            z.n = x.n + y.n
        else:
            z.w = y.w
            z.n = y.n

    @binop()
    def TIMES(z, x, y):
        z.w = x.w + y.w
        z.n = x.n * y.n

GS_monoid = GS.new_monoid(GS.PLUS, GS.one)
GS_semiring = GS.new_semiring(GS_monoid, GS.TIMES)

def shortest_path_FW_early_stop(matrix):
    n = matrix.nrows
    v = Vector.sparse(matrix.type, n)
    D_k = matrix.dup()
    D = matrix.dup()
    D_prev = matrix.dup()
    with GS_semiring:
        while True:
            D_prev = D.dup()
            D_k @= matrix
            D += D_k
            # iseq not working
            if D.iseq(D_prev):
                break
    return D

def shortest_path_FW(matrix):
    n = matrix.nrows
    v = Vector.sparse(matrix.type, n)
    D_k = matrix.dup()
    D = matrix.dup()
    with GS_semiring:
        for _ in range(matrix.nrows):
            D_k @= matrix
            D += D_k
    return D

def algebraic_IPC(Adj, x):
    x = x.astype(np.float64)
    n = Adj.shape[0]
    A = Matrix.sparse(GS, n, n)
    w_s_r = np.float64(0)
    w_v = np.zeros(n, dtype=np.float64)
    # compute relative contribution coefficients (weights)
    for s in range(n):
        for r in range(n):
            if x[r] - x[s] > 0:
                w_s_r += x[r] - x[s]
                w_v[s] += x[r] - x[s]
                w_v[r] += x[r] - x[s]
    for k, v in Adj.items():
        i = k[0]
        j = k[1]
        if i == j:
            A[i,j] = (np.float64('inf'), np.int64(0))
        else:
            A[i,j] = (float(v), np.int64(1))
    D = shortest_path_FW(A)
    #C = Vector.sparse(FP64, n)  
    C = np.zeros(n, dtype=np.float64)
    for v in range(n):
        for s in D.extract_col(v).indices:
            for r in D.extract_row(v).indices:
                if x[r] - x[s] <= 0:
                    continue
                elif s == v or r == v or s == r:
                    continue
                elif np.round(D[s, r][0], DECIMALS) == np.round(D[s, v][0] + D[v, r][0], DECIMALS):
                    w_v_sr = (x[r] - x[s]) / (w_s_r - w_v[v]) 
                    C[v] += np.float64(D[s, v][1] * D[v, r][1]) / np.float64(D[s, r][1]) * w_v_sr
    return C


def algebraic_IPC_with_paths(Adj, x):
    x = x.astype(np.float64)
    n = Adj.shape[0]
    A = Matrix.sparse(GS, n, n)
    A1 = Matrix.sparse(FP64, n, n)
    w_s_r = np.float64(0)
    w_v = np.zeros(n, dtype=np.float64)
    # compute relative contribution coefficients (weights)
    for s in range(n):
        for r in range(n):
            if x[r] - x[s] > 0:
                w_s_r += x[r] - x[s]
                w_v[s] += x[r] - x[s]
                w_v[r] += x[r] - x[s]
    for k, v in Adj.items():
        i = k[0]
        j = k[1]
        if i == j:
            A[i, j] = (np.float64('inf'), np.int64(0))
        else:
            A[i, j] = (v, np.int64(1))
            A1[i,j] = np.float64(v)
    D = shortest_path_FW(A)
    D1 = Matrix.sparse(FP64, n, n)
    for i, j, v in D:
        D1[i,j] = v[0]
    pred = dict()
    with FP64.plus:
        for i in range(n):
            pred[i] = dict()
            d = D1.extract_row(i)
            d[i] = 0
            for j in d.indices:
                col_j = A1.extract_col(j)
                col_j.emult(d, mult_op=FP64.plus, out=col_j)
                #pred_j = list((col_j == d[j]).indices)
                #if pred_j:
                pred[i][j] = []
                for v in col_j.indices:
                    if np.round(col_j[v], DECIMALS) == np.round(d[j], DECIMALS):
                        pred[i][j].append(v)
    C = np.zeros(n, dtype=np.float64)
    paths = dict()
    v_paths = dict()
    for v in range(n):
        for s in D.extract_col(v).indices:
            for r in D.extract_row(v).indices:
                if x[r] - x[s] <= 0:
                    continue
                elif s == v or r == v or s == r:
                    continue
                elif np.round(D[s, r][0], DECIMALS) == np.round(D[s, v][0] + D[v, r][0], DECIMALS):
                    w_v_sr = (x[r] - x[s]) / (w_s_r - w_v[v]) 
                    C[v] += np.float64(D[s, v][1] * D[v, r][1]) / np.float64(D[s, r][1]) * w_v_sr
                    if (s, r) not in paths:
                        paths[(s, r)] = st_paths(pred[s], s, r)
                    if v not in v_paths:
                        v_paths[v] = []
                    #for p in paths[(s, r)]:
                    #    if v in p:
                    #        v_paths[v].append(p)
                    sv_paths = st_paths(pred[s], s, v)
                    vr_paths = st_paths(pred[s], v, r)
                    for p1 in sv_paths:
                        for p2 in vr_paths:
                            v_paths[v].append(p1 + p2[1:])
    return C, v_paths, paths


def st_paths(pred, s, t):
    stack = [[t, 0]]
    top = 0
    paths = []
    while top >= 0:
        node, i = stack[top]
        if node == s:
            paths.append([p for p, n in reversed(stack[:top+1])])
        if node not in pred:
            continue
        if len(pred[node]) > i:
            top += 1
            if top == len(stack):
                stack.append([pred[node][i],0])
            else:
                stack[top] = [pred[node][i],0]
        else:
            stack[top-1][1] += 1
            top -= 1
    return paths


def algebraic_IPC_with_pred(Adj, x):
    x = x.astype(np.float64)
    n = Adj.shape[0]
    A = Matrix.sparse(GS, n, n)
    A1 = Matrix.sparse(FP64, n, n)
    w_s_r = np.float64(0)
    w_v = np.zeros(n, dtype=np.float64)
    # compute relative contribution coefficients (weights)
    for s in range(n):
        for r in range(n):
            if x[r] - x[s] > 0:
                w_s_r += x[r] - x[s]
                w_v[s] += x[r] - x[s]
                w_v[r] += x[r] - x[s]
    for k, v in Adj.items():
        i = k[0]
        j = k[1]
        if i == j:
            A[i, j] = (np.float64('inf'), np.int64(0))
        else:
            A[i, j] = (np.float64(v), np.int64(1))
            A1[i, j] = np.float64(v)
    D = shortest_path_FW(A)
    C = np.zeros(n)
    for v in range(n):
        for s in D.extract_col(v).indices:
            for t in D.extract_row(v).indices:
                if x[t] - x[s] <= 0:
                    continue
                if D[s, t][1] == 0:
                    continue
                if s != v and t!=v and s!=t and D[s, t][0] == D[s, v][0] + D[v, t][0]:
                    w_v_sr = (x[r] - x[s]) / (w_s_r - w_v[v]) 
                    C[v] += D[s, v][1] * D[v, t][1] / D[s, t][1] * w_v_sr
    D1 = Matrix.sparse(FP64, n, n)
    for i, j, v in D:
        D1[i,j] = v[0]
    pred = dict()
    with FP64.plus:
        for i in range(n):
            pred[i] = dict()
            d = D1.extract_row(i)
            d[i] = 0
            for j in d.indices:
                col_j = A1.extract_col(j)
                col_j.emult(d, mult_op=FP64.plus, out=col_j)
                #pred_j = list((col_j == d[j]).indices)
                #if pred_j:
                pred[i][j] = list((col_j == d[j]).indices)
    return C, pred, D1

def example():
    A = Matrix.sparse(GS, 6, 6)
    A[0,1] = (9.0, 1)
    A[0,3] = (3.0, 1)
    A[0,5] = (4.0, 1)
    A[1,2] = (8.0, 1)
    A[3,4] = (6.0, 1)
    A[3,5] = (1.0, 1)
    A[4,2] = (4.0, 1)
    A[1,5] = (7.0, 1)
    A[5,4] = (2.0, 1)
    N = net.Network(notebook=True, directed=True)
    for i, j, v in A:
        N.add_node(i, label=str(i), shape='circle')
        N.add_node(j, label=str(j), shape='circle')
        N.add_edge(i, j, label=str(v[0]))
    N.show('graph.html')

if __name__ == '__main__':
    Adj = dok_matrix((6, 6), dtype=float) 
    Adj[0,1] = 9.0
    Adj[0,3] = 3.0
    Adj[0,5] = 4.0
    Adj[1,2] = 8.0
    Adj[3,4] = 6.0
    Adj[3,5] = 1.0
    Adj[4,2] = 4.0
    Adj[1,5] = 7.0
    Adj[5,4] = 2.0
    states = np.array([0, 0, 0.4, 0, 0, 0])
    C, v_paths = algebraic_IPC_with_paths(Adj, states)
    print(C)
    print(v_paths)

