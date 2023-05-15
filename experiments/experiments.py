import pandas as pd
import numpy as np
import pprint
from datetime import timedelta, date
import time
import warnings
warnings.filterwarnings('ignore')


from teg.schemas import *
from teg.events import *
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
from teg.event_utils import *
from teg.run_experiments import *

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
    'PI_vitals': False, # Use a list of vitals related to PI
    'skip_repeat': False,
    'skip_repeat_intervention': False,
    'quantiles': np.arange(0, 1.01, 0.1),
    'drug_percentile': [40, 70],
    'input_percentile': [40, 85],
    'include_numeric': True,
    'subsequent_adm': False,
    'hadm_limit': 33,
    'NPI_hadm_limit': False,
    'hadm_order': 'DESC',
    'PC_path': False,
    'vis': True,
    'plot': True,
    'first_hadm': True,
    'dbsource': 'metavision', # carevue or False
    'iterations': 5,
    'psm_features':
        [
        'hadm_id',
        'gender', 
        'admission_type',
        'insurance',
        'diagnosis',
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
    all_events = events(conn, event_list, conf, PI_hadms)
    all_events, PI_hadm_stage_t = process_events_PI(all_events, conf)
    run_experiments(patients, all_events, conf, join_rules, fname)


def TEG_PC_Non_PI(event_list, join_rules, conf, fname):
    conn = get_db_connection()
    # PI patients
    PI_df, PI_admissions = get_patient_demography(conn, conf) 
    print('PI Patients', len(PI_admissions))
    PI_hadms = tuple(PI_df['hadm_id'].tolist())
    PI_events = events(conn, event_list, conf, PI_hadms)
    PI_events, PI_hadm_stage_t = process_events_PI(PI_events, conf)
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
    psm = get_psm(df, conf, fname)
    PI_hadms = tuple(psm.matched_ids['hadm_id'].tolist())
    if len(PI_hadm_stage_t) != len(PI_hadms):
        PI_events = events(conn, event_list, conf, PI_hadms)
        PI_events, PI_hadm_stage_t = process_events_PI(PI_events, conf)
    NPI_t = dict()
    NPI_hadms = list()
    for i, row in psm.matched_ids.iterrows():
        if row['hadm_id'] in PI_hadm_stage_t:
            NPI_t[row['matched_ID']] = PI_hadm_stage_t[row['hadm_id']]
            NPI_hadms.append(row['matched_ID'])
    NPI_hadms = tuple(NPI_hadms)
    NPI_events = events(conn, event_list, conf, NPI_hadms)
    NPI_events = process_events_NPI(NPI_events, NPI_t, conf)

    for i in range(conf['iterations']):
        PI_PC_all, PI_PC_nz, PI_PC_P = run_experiments(PI_admissions, PI_events, conf, join_rules, fname + f'_PI_{i}')
        PI_etypes = get_event_types(PI_events, PI_PC_nz)
        PI_etypes_P = get_event_types(PI_events, PI_PC_P)

        NPI_PC_all, NPI_PC_nz, NPI_PC_P = run_experiments(NPI_admissions, NPI_events, conf, join_rules, fname + f'_NPI_{i}')
        NPI_etypes = get_event_types(NPI_events, NPI_PC_nz)
        NPI_etypes_P = get_event_types(NPI_events, NPI_PC_P)
        PI_types, NPI_types, I = intersection_and_differences(PI_etypes_P, NPI_etypes_P)

        PI_events = remove_event_type(PI_events, I) 
        NPI_events = remove_event_type(NPI_events, I) 
        if I == []:
            break
        
if __name__ == "__main__":
    fname_keys = [
        'max_hours',
        'PC_percentile',
        'drug_percentile',
        'input_percentile',
        'skip_repeat',
        'hadm_limit', 
        'min_missing_percent',
        'dbsource']
    fname_LF = 'output/TEG'
    fname_LF += '_' + '_'.join([k + '-' + str(v)
                               for k, v in conf.items() if k in fname_keys])
    fname_LF += '_' + '_'.join([k + '-' + str(v)
                                    for k, v in join_rules.items()
                                    if k in fname_keys])

    fname = 'output/TEG-PI'
    TEG_PC_PI(EVENTS, join_rules, conf, fname)
    #TEG_PC_Non_PI(EVENTS, join_rules, conf, fname)
