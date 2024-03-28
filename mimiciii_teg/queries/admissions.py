import pandas as pd
import numpy as np
import pprint
import copy
import warnings
warnings.filterwarnings('ignore')

from mimiciii_teg.schemas.event_setup import *
from mimiciii_teg.teg.events import *
from mimiciii_teg.vis.plot_centrality import *
from mimiciii_teg.vis.plot_PI_NPI import plot_PI_NPI
from mimiciii_teg.vis.plot_patients import *
from mimiciii_teg.utils.psm import *
from mimiciii_teg.utils.event_utils import group_events_by_hadm_id

def admissions(conn, event_list, join_rules, conf, fname):
    if conf['include_chronic_illness']:
        conf['psm_features'] = conf['psm_features'] + list(CHRONIC_ILLNESS.keys())
    # PI patients
    PI_dff, PI_admissions = get_patient_demography(conn, conf) 
    # Label admissions as PI or Non PI
    PI_dff['PI'] = 1
    print('PI Patients', len(PI_admissions))
    PI_hadms = tuple(PI_dff['hadm_id'].tolist())
    PI_events = events(conn, event_list, conf, PI_hadms)
    PI_events, PI_hadm_stage_t = process_events_PI(PI_events, conf)
    PI_hadms = tuple(list(PI_hadm_stage_t.keys()))
    # Remove invalid admissions
    PI_dff = PI_dff[PI_dff['hadm_id'].isin(list(PI_hadm_stage_t.keys()))]
    # used for PSM
    PI_df = PI_dff[conf['psm_features']]
    # Non PI patients
    conf['PI_sql'] = 'no_PI_stages'
    conf['hadm_limit'] = conf['NPI_hadm_limit']
    NPI_dff, NPI_admissions = get_patient_demography(conn, conf) 

    # to filter NPI admissions by minimum patient events
    NPI_hadms = tuple(NPI_dff['hadm_id'].tolist())
    event_list_npi = [name for name in event_list if 'PI Stage' not in name]
    NPI_events = events(conn, event_list_npi, conf, NPI_hadms)
    hadm_events = group_events_by_hadm_id(NPI_events)
    NPI_hadms = tuple(list(hadm_events.keys()))
    NPI_dff, NPI_admissions = get_patient_demography(conn, conf, NPI_hadms) 

    # Label admissions as PI or Non PI
    NPI_dff['PI'] = 0
    NPI_df = NPI_dff[conf['psm_features']]
    print('Non PI Patients', len(NPI_admissions))
    # Check if PI and NPI admissions intersect
    int_df = pd.merge(PI_df, NPI_df, how ='inner', on =['hadm_id'])
    print('Intersection of PI and NPI patients', len(int_df))
    # Exclude the intersection if exists
    NPI_df = NPI_df.loc[~NPI_df['hadm_id'].isin(PI_df['hadm_id'])]
    I_df = pd.merge(PI_df, NPI_df, how ='inner', on =['hadm_id'])
    print('Intersection of PI and NPI patients', len(I_df))
    # Label admissions as PI or Non PI
    PI_df['PI'] = 1
    NPI_df['PI'] = 0
    df = pd.concat([PI_df, NPI_df])
    # Propensity Score matching
    psm = get_psm(df, fname)
    PI_hadms = tuple(psm.matched_ids['hadm_id'].tolist())
    if len(PI_hadm_stage_t) != len(PI_hadms):
        PI_events = events(conn, event_list, conf, PI_hadms)
        PI_events, PI_hadm_stage_t = process_events_PI(PI_events, conf)
        PI_hadms = tuple(list(PI_hadm_stage_t.keys()))
        PI_dff, PI_admissions = get_patient_demography(conn, conf, PI_hadms) 
        PI_dff['PI'] = 1
    NPI_t = dict()
    NPI_hadms = list()
    PI_NPI_match = dict()
    for i, row in psm.matched_ids.iterrows():
        if row['hadm_id'] in PI_hadm_stage_t:
            NPI_t[row['matched_ID']] = PI_hadm_stage_t[row['hadm_id']]
            NPI_hadms.append(row['matched_ID'])
            PI_NPI_match[row['hadm_id']] = row['matched_ID']
    NPI_hadms = tuple(NPI_hadms)
    # no PI stage events for NPI patients
    event_list_npi = [name for name in event_list if 'PI Stage' not in name]
    NPI_events = events(conn, event_list_npi, conf, NPI_hadms)
    NPI_events = process_events_NPI(NPI_events, NPI_t, conf)
    NPI_dff, NPI_admissions = get_patient_demography(conn, conf, NPI_hadms) 
    # Label admissions as PI or Non PI
    NPI_dff['PI'] = 0
    plot_PI_NPI_patients(PI_admissions,
                         NPI_admissions,
                         PI_dff,
                         NPI_dff,
                         conf,
                         PI_events,
                         NPI_events,
                         fname=f"{fname}_Patients")
    plot_patients(PI_admissions,
                  PI_dff,
                  conf,
                  PI_events,
                  fname=f"{fname}_PI_Patients",
                  c='blue')
    plot_patients(NPI_admissions,
                  NPI_dff,
                  conf,
                  NPI_events,
                  fname=f"{fname}_NPI_Patients",
                  c='red')
    results = dict()
    results['PI_events'] = PI_events
    results['NPI_events'] = NPI_events
    results['PI_admissions'] = PI_admissions
    results['NPI_admissions'] = NPI_admissions
    results['PI_df'] = PI_dff
    results['NPI_df'] = NPI_dff
    results['PI_hadms'] = PI_hadms
    results['NPI_hadms'] = NPI_hadms
    results['PI_hadm_stage_t'] = PI_hadm_stage_t
    pi_groups = group_events_by_hadm_id(PI_events)
    npi_groups = group_events_by_hadm_id(NPI_events)
    pi_hadm_events = sorted([len(hadm_events) for hadm_id, hadm_events in pi_groups.items()])
    npi_hadm_events = sorted([len(hadm_events) for hadm_id, hadm_events in npi_groups.items()])
    print('PI hadm event counts', pi_hadm_events)
    print('NPI hadm event counts', npi_hadm_events)
    return results 
