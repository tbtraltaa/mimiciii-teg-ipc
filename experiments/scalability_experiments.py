import pandas as pd
import numpy as np
import pprint
from datetime import timedelta, date
import time
import networkx as nx
import matplotlib.pyplot as plt
plt.style.use('default')
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

TIME_UNIT = 'sec'
# Experiment configuration
TEG_conf = {
    'include_chronic_illness': True,
    'patient_history': timedelta(weeks=24), # 6 months
    'admission_type': 'EMERGENCY',
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
    'missing_percent': [0, 100], # for mimic extract
    'vitals_agg': 'daily',
    'vitals_X_mean': False,
    'interventions': True,
    'node label': True,
    'edge label': True,
    #'PI_states': {0: 0, 0.5: 0.1, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1},
    'PI_states': {0: 0, 2: 1},
    'PI_exclude_mid_stages': True,
    'PI_daily_max_stage': True,
    'CENTRALITY_time_unit': timedelta(days=0, hours=1), # maximum CENTRALITY per time unit
    'P': [90, 100], # CENTRALITY percentile
    'P_results': [90, 100], # CENTRALITY percentile
    'P_remove': [0, 20],
    'P_patients': [60, 100],
    'ET_CENTRALITY_min_freq': 0,
    'CENTRALITY_path': False,
    'P_max_n': False,
    'path_percentile': [95, 100],
    'PI_sql': 'one', #one, multiple, one_or_multiple, no_PI_stages, no_PI_events
    'PI_only': False, # Delete non PI patients after querying all events
    'PI_as_stage': False, # PI events after stage 0 are considered as stage 1 
    'unique_chartvalue_per_day_sql': False, # Take chart events with distinct values per day
    'unique_chartvalue_per_day': True,
    'has_icustay': 'True',
    'scale_CENTRALITY': False, # scale by max_CENTRALITY
    'Top_n_CENTRALITY': 20,
    'PI_vitals': PI_VITALS, # Use a list of vitals related to PI
    'skip_repeat': False,
    'skip_repeat_intervention': False,
    'quantiles': np.arange(0, 1.01, 0.1),
    'quantile_round': 2,
    'drug_percentile': [40, 60],
    'input_percentile': [40, 80],
    'include_numeric': True,
    'subsequent_adm': False,
    'hadm_limit': False,
    'NPI_hadm_limit': False,
    'hadm_order': 'DESC',
    'n_patient_paths': [1, 2, 3], # n highest CENTRALITY paths of a patient
    'vis': False,
    'plot': False,
    'plot_types_n': 20,
    'CENTRALITY_BS_nnz': 0, # in percentage
    'first_hadm': True,
    'dbsource': 'carevue', # carevue or False
    'iterations': 20,
    'iter_type': 'event_type_CENTRALITY', # event_type_CENTRALITY or event_CENTRALITY
    'CENTRALITY_P_events': False,
    'psm_features':
        [
        'hadm_id',
        'gender', 
        #'admission_type',
        'insurance',
        'los',
        'age',
        'oasis'] 
}

# Event graph configuration
# t_max = [<delta days>, <delta hours>]
t_max = {
    'Admissions': timedelta(days=7, hours=0),
    'Discharges': timedelta(days=7, hours=0),
    'ICU In': timedelta(days=7, hours=0),
    'ICU Out': timedelta(days=7, hours=0),
    'Callout': timedelta(days=7, hours=0),
    'Transfer In': timedelta(days=7, hours=0),
    'Transfer Out': timedelta(days=7, hours=0),
    'CPT': timedelta(days=7, hours=0),
    'Presc': timedelta(days=7, hours=0),
    'Services': timedelta(days=7, hours=0),
    'other': timedelta(days=7, hours=0),
    'diff_type_same_patient': timedelta(days=7, hours=0),
    'PI': timedelta(days=7, hours=0),
    'Braden': timedelta(days=7, hours=0),
    'Input': timedelta(days=7, hours=0),
    'Intervention': timedelta(days=7, hours=0),
    'Vitals/Labs': timedelta(days=7, hours=0),
}

TEG_join_rules = {
    'IDs': EVENT_IDs if not TEG_conf['duration'] else EVENT_IDs + ['duration'],
    "t_min": timedelta(days=0, hours=0, minutes=5),
    "t_max": t_max,
    'w_e_max': 0.3,  # maximum event difference
    # default event difference for different types of events
    'w_e_default': 1,
    'join_by_subject': True,
    'duration': TEG_conf['duration'],
    'duration_similarity': timedelta(days=7),
    'sequential_join': True,
    'max_pi_state': 1,
    'max_pi_stage': 2,
    'include_numeric': True
}


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

    for i in range(10, 40, 10):
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
        if TIME_UNIT == 'min':
            t = min(timer.repeat(repeat=1, number=1)) / 60.0
            print("Time for algebraic IPC", t, 'min' )
        else:
            t = min(timer.repeat(repeat=1, number=1))
            print("Time for algebraic IPC", t, 'sec' )

        algebraic_IPC_time.append(t)
        n_nodes.append(n) 
        m_edges.append(A.count_nonzero())
        # NetworkX graph
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        # non-algebraic IPC using NetworkX
        timer = timeit.Timer("IPC_with_target_nx(G, states=states, weight='weight')", globals=globals())
        if TIME_UNIT == 'min':
            t = min(timer.repeat(repeat=1, number=1)) / 60.0
            print("Time for non-algebraic IPC", t, 'min' )
        else:
            t = min(timer.repeat(repeat=1, number=1))
            print("Time for non-algebraic IPC", t, 'sec' )
        nx_IPC_time.append(t)
    # Plot number of nodes vs IPC time 
    plt.style.use('default')
    plt.figure(figsize=(14, 8), layout='constrained')
    plt.plot(n_nodes, algebraic_IPC_time, label='Algebraic IPC')
    plt.plot(n_nodes, nx_IPC_time, label='Non-algebraic IPC using NetworkX')
    plt.xlabel('Number of Nodes')
    if TIME_UNIT == 'min':
        plt.ylabel('Time ( In Minutes )')
    else:
        plt.ylabel('Time ( In Seconds )')
    plt.title("Inverse Percolation Centrality Scalability")
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}-nodes")
    plt.clf()
    plt.cla()

    # Plot number of edges vs IPC time 
    plt.style.use('default')
    plt.figure(figsize=(14, 8), layout='constrained')
    plt.plot(m_edges, algebraic_IPC_time, label='Algebraic IPC')
    plt.plot(m_edges, nx_IPC_time, label='Non-algebraic IPC using NetworkX')
    plt.xlabel('Number Of Edges')
    if TIME_UNIT == 'min':
        plt.ylabel('Time ( In Minutes )')
    else:
        plt.ylabel('Time ( In Seconds )')
    plt.title("Inverse Percolation Centrality Scalability")
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}-edges")
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

    for i in range(10, 30, 10):
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
        if TIME_UNIT == 'min':
            t = min(timer.repeat(repeat=1, number=1)) / 60.0
            print("Time for algebraic IPC", t, 'min' )
        else:
            t = min(timer.repeat(repeat=1, number=1))
            print("Time for algebraic IPC", t, 'sec' )
        algebraic_IPC_time.append(t)
        n_nodes.append(n) 
        # number of edges
        m_edges.append(A.count_nonzero())
    plt.style.use('default')
    plt.figure(figsize=(14, 8), layout='constrained')
    plt.plot(n_nodes, algebraic_IPC_time, label='Algebraic IPC')
    plt.xlabel('Number of Nodes')
    if TIME_UNIT == 'min':
        plt.ylabel('Time ( In Minutes )')
    else:
        plt.ylabel('Time ( In Seconds )')
    plt.title("Algebraic Inverse Percolation Centrality Scalability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname}-nodes")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.figure(figsize=(14, 8), layout='constrained')
    plt.plot(m_edges, algebraic_IPC_time, label='Algebraic IPC')
    plt.xlabel('Number of Edges')
    if TIME_UNIT == 'min':
        plt.ylabel('Time ( In Minutes )')
    else:
        plt.ylabel('Time ( In Seconds )')
    plt.title("Algebraic Inverse Percolation Centrality Scalability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fname}-edges")
    plt.clf()
    plt.cla()

def algebraic_IPC_n_threads(event_list, join_rules, conf, fname):
    global A, states
    conn = get_db_connection()
    algebraic_IPC_time = []
    # number of threads
    threads = [int(i) for i in range(4, 33, 4)]
    # admissions limit
    conf['hadm_limit'] = 100
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
        if TIME_UNIT == 'min':
            t = min(timer.repeat(repeat=1, number=1)) / 60.0
            print("Time for algebraic IPC", t, 'min' )
        else:
            t = min(timer.repeat(repeat=1, number=1))
            print("Time for algebraic IPC", t, 'sec' )
        algebraic_IPC_time.append(t)
    plt.style.use('default')
    plt.figure(figsize=(14, 8), layout='constrained')
    plt.plot(threads, algebraic_IPC_time, label='Algebraic IPC')
    plt.xlabel('Number Of Threads')
    if TIME_UNIT == 'min':
        plt.ylabel('Time ( In Minutes )')
    else:
        plt.ylabel('Time ( In Seconds )')
    plt.title(f"Algebraic Inverse Percolation Centrality Scalability (n = {n}, m = {m})")
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{fname}")
    plt.clf()
    plt.cla()

        
if __name__ == "__main__":
    '''
    fname = 'output/IPC_vs_algebraic_IPC'
    IPC_vs_algebraic_IPC_scalability(PI_RISK_EVENTS, TEG_join_rules, TEG_conf, fname)
    '''
    fname = 'output/algebraic_IPC_scalability'
    algebraic_IPC_scalability(PI_RISK_EVENTS, TEG_join_rules, TEG_conf, fname)
    '''
    TIME_UNIT = 'min'
    fname = 'output/algebraic_IPC-n-threads'
    algebraic_IPC_n_threads(PI_RISK_EVENTS, TEG_join_rules, TEG_conf, fname)
    '''
