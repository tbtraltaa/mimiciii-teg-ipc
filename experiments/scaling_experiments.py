import pandas as pd
import numpy as np
import pprint
from datetime import timedelta, date
import time
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pygraphblas import *
import timeit

from mimiciii_teg.schemas import *
from mimiciii_teg.schemas.event_setup import *
from mimiciii_teg.schemas.PI_risk_factors import PI_VITALS
from mimiciii_teg.queries.queries import get_patient_demography
from mimiciii_teg.teg.events import *
from mimiciii_teg.teg.eventgraphs import *
from mimiciii_teg.centrality.IPC import IPC_with_target_nx
from mimiciii_teg.centrality.algebraic_IPC import *
from mimiciii_teg.teg.build_graph import *
from TEG_experiments import TEG_conf, TEG_join_rules 


def IPC_vs_algebraic_IPC_scalability(event_list, join_rules, conf, fname):
    '''
    An experiment to compare non-algebraic and algebraic IPC run time.
    '''
    conn = get_db_connection()
    algebraic_IPC_time = []
    nx_IPC_time = []
    n_nodes = []
    m_edges = []
    global G, A, states
    options_set(nthreads=12)

    for i in range(10, 100, 10):
        # number of patients
        conf['hadm_limit'] = i
        PI_df, admissions = get_patient_demography(conn, conf) 
        print('Patients', len(admissions))
        PI_hadms = tuple(PI_df['hadm_id'].tolist())
        all_events = events(conn, event_list, conf, PI_hadms)
        all_events, PI_hadm_stage_t = process_events_PI(all_events, conf)
        # number of nodes
        n = len(all_events)
        # adjacency matrix
        A, interconnection = build_eventgraph(admissions, all_events, join_rules)
        states = np.zeros(n)
        # Percolation states
        for e in all_events:
            states[e['i']] = e['pi_state']
        # algebraic IPC
        timer = timeit.Timer('algebraic_IPC(A, states=states)', globals=globals())
        t = min(timer.repeat(repeat=10, number=1))
        print("Time for algebraic IPC", t, 'sec' )
        algebraic_IPC_time.append(t)
        n_nodes.append(n) 
        m_edges.append(A.count_nonzero())
        # NetworkX graph
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        # non-algebraic IPC using NetworkX
        timer = timeit.Timer("IPC_with_target_nx(G, states=states, weight='weight')", globals=globals())
        t = min(timer.repeat(repeat=10, number=1))
        print("Time for non-algebraic IPC", t, 'sec' )
        nx_IPC_time.append(t)
    # Plot number of nodes vs IPC time 
    plt.figure(figsize=(14, 8), layout='constrained')
    plt.plot(n_nodes, algebraic_IPC_time, label='Algebraic IPC')
    plt.plot(n_nodes, nx_IPC_time, label='Non-algebraic IPC using NetworkX')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time ( in seconds )')
    plt.title("Inverse Percolation Centrality Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname}-Scaling_experiment-nodes")
    plt.clf()
    plt.cla()

    # Plot number of edges vs IPC time 
    plt.figure(figsize=(14, 8), layout='constrained')
    plt.figure(figsize=(14, 8), layout='constrained')
    plt.plot(m_edges, algebraic_IPC_time, label='Algebraic IPC')
    plt.plot(m_edges, nx_IPC_time, label='Non-algebraic IPC using NetworkX')
    plt.xlabel('Number Of Edges')
    plt.ylabel('Time ( In Seconds )')
    plt.title("Percolation Centrality Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname}-Scaling_experiment-edges")
    plt.clf()
    plt.cla()

def algebraic_IPC_scalability(event_list, join_rules, conf, fname):
    '''
    An experiment to show the scalability of algebraic IPC 
    '''
    conn = get_db_connection()
    algebraic_IPC_time = []
    n_nodes = []
    m_edges = []
    global A, states
    options_set(nthreads=12)

    for i in range(50, 300, 50):
        # limit to admissions
        conf['hadm_limit'] = i
        PI_df, admissions = get_patient_demography(conn, conf) 
        print('Patients', len(admissions))
        PI_hadms = tuple(PI_df['hadm_id'].tolist())
        all_events = events(conn, event_list, conf, PI_hadms)
        all_events, PI_hadm_stage_t = process_events_PI(all_events, conf)
        # number of nodes
        n = len(all_events)
        # adjacency matrix
        A, interconnection = build_eventgraph(admissions, all_events, join_rules)
        states = np.zeros(n)
        # percolation states
        for e in all_events:
            states[e['i']] = e['pi_state']
        # algebraic IPC
        timer = timeit.Timer('algebraic_IPC(A, states=states)', globals=globals())
        t = min(timer.repeat(repeat=10, number=1))
        print("Time for algebraic IPC", t, 'sec' )
        algebraic_IPC_time.append(t)
        n_nodes.append(n) 
        # number of edges
        m_edges.append(A.count_nonzero())
    plt.figure(figsize=(14, 8), layout='constrained')
    plt.plot(n_nodes, algebraic_IPC_time, label='Algebraic IPC')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time ( in sec )')
    plt.title("Inverse Percolation Centrality Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname}-Scaling_experiment-nodes")
    plt.clf()
    plt.cla()

    plt.figure(figsize=(14, 8), layout='constrained')
    plt.plot(m_edges, algebraic_IPC_time, label='Algebraic IPC')
    plt.xlabel('Number of Edges')
    plt.ylabel('Time ( in sec )')
    plt.title("Inverse Percolation Centrality Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname}-Scaling_experiment-edges")
    plt.clf()
    plt.cla()

def algebraic_IPC_n_threads(event_list, join_rules, conf, fname):
    global A, states
    conn = get_db_connection()
    algebraic_IPC_time = []
    # number of threads
    threads = [int(i) for i in range(4, 33, 4)]
    # admissions limit
    conf['hadm_limit'] = False
    PI_df, admissions = get_patient_demography(conn, conf) 
    print('Patients', len(admissions))
    PI_hadms = tuple(PI_df['hadm_id'].tolist())
    all_events = events(conn, event_list, conf, PI_hadms)
    all_events, PI_hadm_stage_t = process_events_PI(all_events, conf)
    # number of nodes
    n = len(all_events)
    # adjacency matrix
    A, interconnection = build_eventgraph(admissions, all_events, join_rules)
    # number of edges
    m = A.count_nonzero()
    # percolation states
    states = np.zeros(n)
    for e in all_events:
        states[e['i']] = e['pi_state']
    for i in threads:
        # set the number of threads for GraphBLAS
        # algebraic IPC
        print('i', i)
        #print(options_get())
        options_set(nthreads=i)
        #print(options_get())
        #print("Globals", globals())
        #timer = timeit.Timer('algebraic_IPC(A, states=states)', setup='options_set(nthreads=i)', globals=globals())
        timer = timeit.Timer('algebraic_IPC(A, states=states)', globals=globals())
        t = min(timer.repeat(repeat=10, number=1))
        print("Time for algebraic IPC", t, 'sec' )
        algebraic_IPC_time.append(t)
    plt.figure(figsize=(14, 8), layout='constrained')
    plt.plot(threads, algebraic_IPC_time, label='Algebraic IPC')
    plt.xlabel('Number Of Threads')
    plt.ylabel('Time ( in sec )')
    plt.title(f"Inverse Percolation Centrality Scaling (n = {n}, m = {m})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname}-Scaling_experiment-nodes")
    plt.clf()
    plt.cla()

        
if __name__ == "__main__":
    fname = 'output/IPC_vs_algebraic_IPC'
    IPC_vs_algebraic_IPC_scalability(PI_RISK_EVENTS, TEG_join_rules, TEG_conf, fname)
    fname = 'output/algebraic_IPC_scalability'
    algebraic_IPC_scalability(PI_RISK_EVENTS, TEG_join_rules, TEG_conf, fname)
    fname = 'output/algebraic_IPC-n-threads'
    algebraic_IPC_n_threads(PI_RISK_EVENTS, TEG_join_rules, TEG_conf, fname)
