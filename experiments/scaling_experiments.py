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


from teg.schemas import *
from teg.event_setup import *
from teg.events import *
from teg.eventgraphs import *
from teg.percolation import PC_with_target_nx
from teg.apercolation import *
from teg.build_graph import *

# Experiment configuration
conf = {
    'duration': False,
    'max_hours': 336,
    'min_los_hours': 24,
    #'max_hours': 168,
    'min_age': 15,
    'max_age': 89,
    'age_interval': 5, # in years, for patients
    'starttime': False,
    #'starttime': '2143-01-14',
    #'endtime': '2143-01-21',
    #'endtime': '2143-02-14',
    'endtime': False,
    'min_missing_percent': 0, # for mimic extract
    'vitals_agg': 'daily',
    'vitals_X_mean': False,
    'interventions': True,
    'node label': True,
    'edge label': True,
    #'PI_states': {0: 0, 0.5: 0.1, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1},
    'PI_states': {0: 0, 2: 1},
    'PI_exclude_mid_stages': True,
    'PC_time_unit': timedelta(days=1, hours=0), # maximum PC per time unit
    'PC_percentile': [95, 100],
    'PC_percentile_max_n': False,
    'path_percentile': [95, 100],
    'PI_sql': 'one', #multiple, one_or_multiple, no_PI_stages, no_PI_events
    'PI_only': False, # Delete non PI patients after querying all events
    'PI_as_stage': False, # PI events after stage 0 are considered as stage 1 
    'unique_chartvalue_per_day_sql': False, # Take chart events with distinct values per day
    'unique_chartvalue_per_day': True,
    'has_icustay': 'True',
    'scale_PC': True, # scale by max_PC
    'Top_n_PC': 20,
    'PI_vitals': True, # Use a list of vitals related to PI
    'skip_repeat': True,
    'skip_repeat_intervention': False,
    'quantiles': np.arange(0, 1.01, 0.1),
    'drug_percentile': [40, 70],
    'input_percentile': [20, 90],
    'include_numeric': True,
    'subsequent_adm': False,
    'hadm_limit': False,
    'NPI_hadm_limit': False,
    'hadm_order': 'DESC',
    'PC_path': False,
    'n_patient_paths': [1, 3, 5], # n highest PC paths of a patient
    'vis': False,
    'plot': False,
    'PC_BS_nnz': 0.1, # in percentage
    'first_hadm': True,
    'dbsource': 'metavision', # carevue or False
    'iterations': 10,
    'PC_P_events': False,
    'psm_features':
        [
        'hadm_id',
        'gender', 
        'admission_type',
        'insurance',
        'los',
        'age',
        'oasis']
}

# Event graph configuration
# t_max = [<delta days>, <delta hours>]
t_max = {
    'Admissions': timedelta(days=2, hours=0),
    'Discharges': timedelta(days=2, hours=0),
    'ICU In': timedelta(days=2, hours=0),
    'ICU Out': timedelta(days=2, hours=0),
    'Callout': timedelta(days=2, hours=0),
    'Transfer In': timedelta(days=2, hours=0),
    'Transfer Out': timedelta(days=2, hours=0),
    'CPT': timedelta(days=2, hours=0),
    'Presc': timedelta(days=2, hours=0),
    'Services': timedelta(days=2, hours=0),
    'other': timedelta(days=2, hours=0),
    'diff_type_same_patient': timedelta(days=2, hours=0),
    'PI': timedelta(days=2, hours=0),
    'Braden': timedelta(days=2, hours=0),
    'Input': timedelta(days=2, hours=0),
    'Intervention': timedelta(days=2, hours=0),
    'Vitals/Labs': timedelta(days=2, hours=0),
}

join_rules = {
    'IDs': EVENT_IDs if not conf['duration'] else EVENT_IDs + ['duration'],
    "t_min": timedelta(days=0, hours=0, minutes=5),
    "t_max": t_max,
    'w_e_max': 0.3,  # maximum event difference
    # default event difference for different types of events
    'w_e_default': 1,
    'join_by_subject': True,
    'duration': conf['duration'],
    'duration_similarity': timedelta(days=2),
    'sequential_join': True,
    'max_pi_state': 1,
    'max_pi_stage': 2,
    'include_numeric': True
}


def TEG_PI_PC_APC(event_list, join_rules, conf, fname):
    '''
    An experiment to compare non-algebraic and algebraic PC run time.
    '''
    conn = get_db_connection()
    APC_time = []
    PC_nx_time = []
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
        # algebraic PC
        timer = timeit.Timer('algebraic_PC(A, states=states)', globals=globals())
        t = min(timer.repeat(repeat=10, number=1))
        print("Time for algebraic PC", t, 'sec' )
        APC_time.append(t)
        n_nodes.append(n) 
        m_edges.append(A.count_nonzero())
        # NetworkX graph
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        # non-algebraic PC using NetworkX
        timer = timeit.Timer("PC_with_target_nx(G, states=states, weight='weight')", globals=globals())
        t = min(timer.repeat(repeat=10, number=1))
        print("Time for non-algebraic PC", t, 'sec' )
        PC_nx_time.append(t)
    # Plot number of nodes vs PC time 
    plt.figure(figsize=(14, 8), layout='constrained')
    plt.plot(n_nodes, APC_time, label='Algebraic PC')
    plt.plot(n_nodes, PC_nx_time, label='Non-algebraic PC using NetworkX')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time ( in seconds )')
    plt.title("Percolation Centrality Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname}-Scaling_experiment-nodes")
    plt.clf()
    plt.cla()

    # Plot number of edges vs PC time 
    plt.figure(figsize=(14, 8), layout='constrained')
    plt.figure(figsize=(14, 8), layout='constrained')
    plt.plot(m_edges, APC_time, label='Algebraic PC')
    plt.plot(m_edges, PC_nx_time, label='Non-algebraic PC using NetworkX')
    plt.xlabel('Number of Edges')
    plt.ylabel('Time ( in seconds )')
    plt.title("Percolation Centrality Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname}-Scaling_experiment-edges")
    plt.clf()
    plt.cla()

def TEG_PI_APC(event_list, join_rules, conf, fname):
    '''
    An experiment to show the scalability of algebraic PC 
    '''
    conn = get_db_connection()
    APC_time = []
    n_nodes = []
    m_edges = []
    global A, states
    options_set(nthreads=12)
    for i in range(10, 100, 10):
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
        # algebraic PC
        timer = timeit.Timer('algebraic_PC(A, states=states)', globals=globals())
        t = min(timer.repeat(repeat=10, number=1))
        print("Time for algebraic PC", t, 'sec' )
        APC_time.append(t)
        n_nodes.append(n) 
        # number of edges
        m_edges.append(A.count_nonzero())
    plt.figure(figsize=(14, 8), layout='constrained')
    plt.plot(n_nodes, APC_time, label='Algebraic PC')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time ( in sec )')
    plt.title("Percolation Centrality Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname}-Scaling_experiment-nodes")
    plt.clf()
    plt.cla()

    plt.figure(figsize=(14, 8), layout='constrained')
    plt.plot(m_edges, APC_time, label='Algebraic PC')
    plt.xlabel('Number of Edges')
    plt.ylabel('Time ( in sec )')
    plt.title("Percolation Centrality Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname}-Scaling_experiment-edges")
    plt.clf()
    plt.cla()

def TEG_PI_APC_n_threads(event_list, join_rules, conf, fname):
    global A, states, i
    conn = get_db_connection()
    APC_time = []
    # number of threads
    threads = [i for i in range(4, 33, 4)]
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
        # algebraic PC
        timer = timeit.Timer('algebraic_PC(A, states=states)', setup='options_set(nthreads=i)', globals=globals())
        t = min(timer.repeat(repeat=10, number=1))
        print("Time for algebraic PC", t, 'sec' )
        APC_time.append(t)
    plt.figure(figsize=(14, 8), layout='constrained')
    plt.plot(threads, APC_time, label='Algebraic PC')
    plt.xlabel('Number of Threads')
    plt.ylabel('Time ( in sec )')
    plt.title(f"Percolation Centrality Scaling (n = {n}, m = {m})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname}-Scaling_experiment-nodes")
    plt.clf()
    plt.cla()

        
if __name__ == "__main__":
    #fname = 'output/TEG-PI-PC-APC'
    #TEG_PI_PC_APC(EVENTS_INCLUDED, join_rules, conf, fname)
    #fname = 'output/TEG-PI-APC'
    #TEG_PI_APC(EVENTS_INCLUDED, join_rules, conf, fname)
    fname = 'output/TEG-PI-APC-n-threads'
    TEG_PI_APC_n_threads(EVENTS_INCLUDED, join_rules, conf, fname)
