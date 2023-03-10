from pygraphblas import *
from pygraphblas import lib
from pygraphblas.types import Type, binop
from pygraphblas.gviz import draw, draw_vis, draw_vector, draw_matrix
import numpy as np
from pyvis import network as net
from scipy.sparse import dok_matrix

options_set(nthreads=8)

class GS(Type):

    _base_name = "UDT"
    _numpy_t = None
    members = ['double w', 'uint64_t n']
    one = (float('inf'), 0)

    @binop(boolean=True)
    def EQ(z, x, y):
        if x.w == y.w and x.n == y.n:
            z = True
        else:
            z = False

    @binop()
    def PLUS(z, x, y):
        if x.w < y.w:
            z.w = x.w
            z.n = x.n
        elif x.w == y.w:
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

def algebraic_PC(Adj, states, normalize=False):
    n = Adj.shape[0]
    A = Matrix.sparse(GS, n, n)
    S = 0.0
    S_vt = dict()
    S_sv = dict()
    for v in range(n):
        deltas = states - states[v]
        S_vt[v] = np.sum(deltas[deltas > 0])
        deltas = states * (-1) + states[v]
        S_sv[v] = np.sum(deltas[deltas > 0])
        S += S_vt[v]
    for k, v in Adj.items():
        i = k[0]
        j = k[1]
        if i == j:
            A[i,j] = (float('inf'), 0)
        else:
            A[i,j] = (v, 1)
    D = shortest_path_FW(A)
    PC = Vector.sparse(FP64, n)  
    for v in range(n):
        S_exclude_v = S - S_sv[v] - S_vt[v]
        for s in D.extract_col(v).indices:
            for t in D.extract_row(v).indices:
                if s != v and t!=v and s!=t and D[s, t][0] == D[s, v][0] + D[v, t][0]:
                    delta = states[t] - states[s]
                    if delta <= 0:
                        continue
                    w = delta / S_exclude_v 
                    if v in PC.indices:
                        PC[v] += D[s, v][1] * D[v, t][1] / D[s, t][1] * w
                    else:
                        PC[v] = D[s, v][1] * D[v, t][1] / D[s, t][1] * w
    return PC

def algebraic_PC_with_paths(Adj, states, normalize=False):
    n = Adj.shape[0]
    A = Matrix.sparse(GS, n, n)
    A1 = Matrix.sparse(FP64, n, n)
    S = 0.0
    S_vt = dict()
    S_sv = dict()
    for v in range(n):
        deltas = states - states[v]
        S_vt[v] = np.sum(deltas[deltas > 0])
        deltas = states * (-1) + states[v]
        S_sv[v] = np.sum(deltas[deltas > 0])
        S += S_vt[v]
    for k, v in Adj.items():
        i = k[0]
        j = k[1]
        if i == j:
            A[i,j] = (float('inf'), 0)
        else:
            A[i,j] = (v, 1)
            A1[i,j] = v
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
                pred[i][j] = list((col_j == d[j]).indices)
    #PC = Vector.sparse(FP64, n)
    PC = np.zeros(n)
    paths = dict()
    v_paths = dict()
    V = set()
    for v in range(n):
        S_exclude_v = S - S_sv[v] - S_vt[v]
        v_paths[v] = []
        for s in D.extract_col(v).indices:
            for t in D.extract_row(v).indices:
                if s != v and t!=v and s!=t and D[s, t][0] == D[s, v][0] + D[v, t][0]:
                    delta = states[t] - states[s]
                    if delta <= 0:
                        continue
                    w = delta / S_exclude_v
                    if (s, t) not in paths:
                        paths[(s,t)] = st_paths(pred[s], s, t)
                        for p in paths[(s, t)]:
                            V.update(p)
                    for p in paths[(s, t)]:
                        if v in p:
                            v_paths[v].append(p)
                    PC[v] += D[s, v][1] * D[v, t][1] / D[s, t][1] * w
    return PC, list(V), v_paths, paths

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
    PC, v_paths = algebraic_PC_with_paths(Adj, states)
    print(PC)
    print(v_paths)

