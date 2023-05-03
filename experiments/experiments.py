import pandas as pd
import numpy as np
import pprint
from datetime import timedelta
import time
import warnings
warnings.filterwarnings('ignore')


from teg.schemas import *
from teg.mimic_events import *
from teg.eventgraphs import *
from teg.percolation import PC_with_target
from teg.apercolation import *
from teg.PC_utils import *
from teg.graph_vis import *
from teg.build_graph import *
from teg.paths import *
from teg.plot import *
from teg.psm import *
from teg.pca import *

# Experiment configuration
conf = {
    'duration': False,
    'max_hours': 336,
    #'max_hours': 168,
    'min_age': 15,
    'max_age': 89,
    'age_interval': 5, # in years, for patients
    'starttime': False,
    #'starttime': '2143-01-14',
    #'endtime': '2143-01-21',
    #'endtime': '2143-02-14',
    'endtime': False,
    'min_missing_percent': 20, # for mimic extract
    'vitals_agg': 'daily',
    'vitals_X_mean': False,
    'interventions': True,
    'node label': True,
    'edge label': True,
    #'PI_states': {0: 0, 0.5: 0.1, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1},
    'PI_states': {0: 0, 1: 1},
    'PC_percentile': [97, 100],
    'path_percentile': [95, 100],
    'PI_sql': 'one', #multiple, one_or_multiple, no_PI_stages, no_PI_events
    'PI_only': False, # Delete non PI patients after querying all events
    'admittime_start':'2143-01-14',
    'admittime_end': '2143-02-14',
    'PI_as_stage': False, # PI events after stage 0 are considered as stage 1 
    'unique_chartvalue_per_day_sql': False, # Take chart events with distinct values per day
    'unique_chartvalue_per_day': True,
    'has_icustay': 'True',
    'scale_PC': True, # scale by max_PC
    'Top_n_PC': 20,
    'PI_vitals': False, # Use a list of vitals related to PI
    'skip_repeat': False,
    'quantiles': np.arange(0, 1.01, 0.1),
    'drug_percentile': [40, 60],
    'input_percentile': [40, 90],
    'include_numeric': True,
    'subsequent_adm': False,
    'hadm_limit': 5,
    'NPI_hadm_limit': False,
    'hadm_order': 'DESC',
    'vis': False,
    'first_hadm': True,
    'dbsource': 'metavision', # carevue or False
    'psm_features':
        [
        'hadm_id',
        'gender', 
        'admission_type',
        'insurance',
        'diagnosis',
        'los',
        'age']
}

# Event graph configuration
# t_max = [<delta days>, <delta hours>]
t_max = {
    'Admissions': timedelta(days=2, hours=0),
    'Discharges': timedelta(days=2, hours=0),
    'Icu In': timedelta(days=2, hours=0),
    'Icu_Out': timedelta(days=2, hours=0),
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
    'max_pi_stage':1,
    'include_numeric': True
}


def TEG_PC_PI(event_list, join_rules, conf, fname):
    conn = get_db_connection()
    PI_df, patients = get_patient_demography(conn, conf) 
    print('Patients', len(patients))
    PI_hadms = tuple(PI_df['hadm_id'].tolist())
    all_events = mimic_events(conn, event_list, conf, PI_hadms)
    events, PI_hadm_stage_t = process_events_PI(all_events, conf)
    n = len(events)
    A = build_eventgraph(patients, events, join_rules)
    states = np.zeros(n)
    for e in events:
        states[e['i']] = e['pi_state']
    if not conf['vis']:
        start = time.time()
        PC_values = algebraic_PC(A, states=states)
        print("Time for PC without paths ", float(time.time() - start)/60.0, 'min' )
    else:
        start = time.time()
        #PC_values, pred, D = algebraic_PC_with_pred(A, states=states)
        #print("Time for PC with pred", float(time.time() - start)/60.0)
        #V, v_paths, paths = PC_paths(D, pred, states)
        #print("Paths")
        PC_values, V, v_paths, paths = algebraic_PC_with_paths(A, states=states)
        print('Algebraic PC time with paths', float(time.time() - start)/60.0, 'min')
        # Check if algebraic PC match PC from networkx
        #print(PC_values)
        b = np.sort(np.nonzero(PC_values)[0])
       # print(b)
    '''
    start = time.time()
    P, pred, D = algebraic_PC_with_pred(A, states=states)
    print("Time for PC with pred", float(time.time() - start)/60.0)
    start = time.time()
    V, v_paths, paths = PC_paths(D, pred, states)
    print("Compute paths", float(time.time() - start)/60.0)
    start = time.time()
    PC_values, V, v_paths = algebraic_PC_with_paths_v1(A, states=states)
    print("Time for PC with pred", float(time.time() - start)/60.0)
    '''
    '''
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    print(nx.is_directed_acyclic_graph(G))
    start = time.time()
    PC, V, v_paths, paths = PC_with_target(G, states=states, weight='weight')
    PC_values = np.array(list(PC.values()))
    print(PC)
    print('PC time based on networkx', float(time.time() - start)/60.0)
    '''
    '''
    # Check if algebraic PC matches PC computed using NetworkX
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    print(nx.is_directed_acyclic_graph(G))
    start = time.time()
    PC, V_nx, v_paths_nx, paths_nx = PC_with_target(G, states=states, weight='weight')
    print(PC)
    print('PC time based on networkx', float(time.time() - start)/60.0)
    a = np.sort(np.nonzero(list(PC.values()))[0])
    print(a)
    print('Algebraic PC nodes match that from networkx:', np.all(a==b))
    print('PC values match:', np.all(list(PC.values())==PC_values))
    print('V match:', np.all(sorted(V)==sorted(V_nx)))
    print('v_paths match:',v_paths==paths_nx)
    print('paths match:', paths==paths_nx)
    '''
    PC_all, PC_nz, PC_P = process_PC_values(PC_values, conf) 
    plot_PC(events, PC_nz, conf, nbins=30)
    plot_PC(events, PC_P, conf, conf['PC_percentile'], nbins=10)
    if conf['vis']:
        visualize(patients, events, A, V, PC_all, PC_P, v_paths, paths, conf, join_rules, fname+'Paths')
        simple_visualization(A, events, patients, PC_all, PC_P, conf, join_rules, fname+'Simple')

def TEG_PC_Non_PI(event_list, join_rules, conf, fname):
    conn = get_db_connection()
    # PI patients
    PI_df, PI_admissions = get_patient_demography(conn, conf) 
    print('PI Patients', len(PI_admissions))
    PI_hadms = tuple(PI_df['hadm_id'].tolist())
    all_events = mimic_events(conn, event_list, conf, PI_hadms)
    PI_events, PI_hadm_stage_t = process_events_PI(all_events, conf)
    # Remove invalid admissions
    PI_df = PI_df[PI_df['hadm_id'].isin(list(PI_hadm_stage_t.keys()))]
    PI_df = PI_df[conf['psm_features']]
    print(PI_df.columns)
    # Non PI patients
    conf['PI_sql'] = 'no_PI_events'
    conf['hadm_limit'] = conf['NPI_hadm_limit']
    NPI_df, NPI_admissions = get_patient_demography(conn, conf) 
    NPI_df = NPI_df[conf['psm_features']]
    print('Non PI Patients', len(NPI_admissions))
    # Check if PI and NPI admissions intersect
    int_df = pd.merge(PI_df, NPI_df, how ='inner', on =['hadm_id'])
    print('Intersection of PI and NPI patients', len(int_df))
    print(int_df)
    # Exclude the intersection if exists
    NPI_df = NPI_df.loc[~NPI_df['hadm_id'].isin(PI_df['hadm_id'])]
    int_df = pd.merge(PI_df, NPI_df, how ='inner', on =['hadm_id'])
    print(len(int_df))
    print(int_df)
    # Label admissions as PI or Non PI
    NPI_df['PI'] = 0
    PI_df['PI'] = 1
    df = pd.concat([PI_df, NPI_df])
    # Propensity Score matrching
    psm = get_psm(df, conf)
    PI_hadms = tuple(psm.matched_ids['hadm_id'].tolist())
    if len(PI_hadm_stage_t) != len(PI_hadms):
        all_events = mimic_events(conn, event_list, conf, PI_hadms)
        PI_events, PI_hadm_stage_t = process_events_PI(all_events, conf)
    n = len(PI_events)
    A = build_eventgraph(PI_admissions, PI_events, join_rules)
    states = np.zeros(n)
    for e in PI_events:
        states[e['i']] = e['pi_state']
    if not conf['vis']:
        start = time.time()
        PC_values = algebraic_PC(A, states=states)
        print("Time for PC without paths ", float(time.time() - start)/60.0, 'min' )
    else:
        start = time.time()
        PC_values, V, v_paths, paths = algebraic_PC_with_paths(A, states=states)
        print('Algebraic PC time with paths', float(time.time() - start)/60.0, 'min')
        b = np.sort(np.nonzero(PC_values)[0])
    PI_PC_all, PI_PC_nz, PI_PC_P = process_PC_values(PC_values, conf)
    plot_PC(PI_events, PI_PC_nz, conf, nbins=30)
    PI_etypes = set()
    for i, val in PI_PC_nz.items():
            PI_etypes.add(PI_events[i]['type'])
    plot_PC(PI_events, PI_PC_P, conf, conf['PC_percentile'], nbins=10)
    PI_etypes_P = set()
    for i, val in PI_PC_P.items():
        PI_etypes_P.add(PI_events[i]['type'])
    if conf['vis']:
        visualize(PI_admissions, PI_events, A, V, PI_PC_all, PI_PC_P, v_paths, paths, conf, fname)
    else:
        simple_visualization(A, PI_events, PI_admissions, PI_PC_all, PI_PC_P, conf, join_rules, fname+'PI')
    # handle admissions with higher or lower PI stages
    # handle subsequent admissions
    NPI_t = dict()
    NPI_hadms = list()
    for i, row in psm.matched_ids.iterrows():
        if row['hadm_id'] in PI_hadm_stage_t:
            NPI_t[row['matched_ID']] = PI_hadm_stage_t[row['hadm_id']]
            NPI_hadms.append(row['matched_ID'])
    NPI_hadms = tuple(NPI_hadms)
    all_events = mimic_events(conn, event_list, conf, NPI_hadms)
    NPI_events = process_events_NPI(all_events, NPI_t, conf)
    n = len(NPI_events)
    A = build_eventgraph(NPI_admissions, NPI_events, join_rules)
    states = np.zeros(n)
    for e in NPI_events:
        states[e['i']] = e['pi_state']
    if not conf['vis']:
        start = time.time()
        PC_values = algebraic_PC(A, states=states)
        print("Time for PC without paths ", float(time.time() - start)/60.0, 'min' )
    else:
        start = time.time()
        PC_values, V, v_paths, paths = algebraic_PC_with_paths(A, states=states)
        print('Algebraic PC time with paths', float(time.time() - start)/60.0, 'min')
        b = np.sort(np.nonzero(PC_values)[0])
    NPI_PC_all, NPI_PC_nz, NPI_PC_P = process_PC_values(PC_values, conf)
    plot_PC(NPI_events, NPI_PC_nz, conf, nbins=30)
    NPI_etypes = set()
    for i, val in NPI_PC_nz.items():
        NPI_etypes.add(NPI_events[i]['type'])
    plot_PC(NPI_events, NPI_PC_P, conf, conf['PC_percentile'], nbins=10)
    NPI_etypes_P = set()
    for i, val in NPI_PC_P.items():
        NPI_etypes_P.add(NPI_events[i]['type'])
    '''
    I = PI_etypes & NPI_etypes
    PI_etypes_diff = PI_etypes - I
    NPI_etypes_diff = NPI_etypes - I
    print("PI differences, Intersection and NPI difference")
    pprint.pprint(PI_etypes_diff)
    pprint.pprint(I)
    pprint.pprint(NPI_etypes_diff)
    '''
    I_P = PI_etypes_P & NPI_etypes_P
    PI_etypes_P_diff = PI_etypes_P - I_P
    NPI_etypes_P_diff = NPI_etypes_P - I_P
    print("With percentile, PI differences, Intersection and NPI difference")
    pprint.pprint(PI_etypes_P_diff)
    pprint.pprint(I_P)
    pprint.pprint(NPI_etypes_P_diff)
    if conf['vis']:
        visualize(NPI_admissions, NPI_events, A, V, NPI_PC_all, NPI_PC_P, v_paths, paths, conf, fname)
    else:
        simple_visualization(A, NPI_events, NPI_admissions, NPI_PC_all, NPI_PC_P, conf, join_rules, fname+'NPI')

if __name__ == "__main__":
    fname_keys = [
        'max_hours',
        'starttime',
        'endtime',
        'PC_percentile',
        'drug_percentile',
        'input_percentile',
        'skip_repeat',
        'PI_only_sql',
        'hadm_limit', 
        'min_missing_percent',
        'dbsource']
    fname_LF = 'output/TEG'
    fname_LF += '_' + '_'.join([k + '-' + str(v)
                               for k, v in conf.items() if k in fname_keys])
    fname_LF += '_' + '_'.join([k + '-' + str(v)
                                    for k, v in join_rules.items()
                                    if k in fname_keys])

    TEG_PC_PI(EVENTS, join_rules, conf, fname_LF)
    #TEG_PC_Non_PI(EVENTS, join_rules, conf, fname_LF)
