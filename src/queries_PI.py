from schemas_PI import *
import numpy as np
import pandas as pd
import sys
import psycopg2
import pprint
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')


PI_EVENT_NAMES = ['PI stage', 'PI site']
# an event query : [<id>, <event_type>, <time>, **<event_attributes>$]
# an event dict: [{col_name: col_value, ...}]


def get_all_PI_events(conn, event_name, conf):
    '''
    Returns PI events within the given time window.
    '''
    schema = 'mimiciii'
    table = 'chartevents'
    time_col = 'charttime'
    label_CV, ignored_values_CV = PI_EVENTS_CV[event_name]
    label_MV, ignored_values_MV = PI_EVENTS_MV[event_name]

    cols = f"CONCAT(tb.subject_id, '-', tb.hadm_id) as id"
    cols += f", '{event_name}' as type, tb.{time_col} - a.admittime as t"
    cols += f'tb.value, d.label as itemid_label'
    table = f'{schema}.{table} tb INNER JOIN {schema}.admissions a'
    table += f' ON tb.hadm_id = a.hadm_id'
    table += f' INNER JOIN {schema}.patients p'
    table += f' ON tb.subject_id = p.subject_id'
    table += f' INNER JOIN {schema}.d_items d'
    table += f' ON tb.itemid = d.itemid'
    where = 'tb.hadm_id is NOT NULL'
    where += " AND a.diagnosis != 'NEWBORN'"
    where += f" AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob))" + \
        f">= {conf['min_age']}"
    where += f" AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob))" + \
        f"<= {conf['max_age']}"
    where += f" AND tb.{time_col} - a.admittime <= '{conf['max_hours']} hours'"
    if conf['starttime'] and conf['endtime']:
        where += f" AND tb.{time_col} >= '{conf['starttime']}'"
        where += f" AND tb.{time_col} < '{conf['endtime']}'"
    # Some events occur before the admisson date, but have the correct hadm_id.
    # Those events are ignored.
    where += f' AND tb.{time_col} >= a.admittime'
    where += f' AND tb.{time_col} <= a.dischtime'
    where += f" AND (d.label similar to '{label_CV}'" + \
        f"OR d.label SIMILAR TO '{label_MV}')"
    where += f' AND tb.value is NOT NULL'
    for value in ignored_values_CV:
        where += f" AND tb.value not similar to '{value}'"
    for value in ignored_values_MV:
        where += f" AND tb.value not similar to '{value}'"
    order_by = f'ORDER BY t ASC'
    query = f"SELECT {cols} FROM {table} WHERE {where} {order_by}"
    df = pd.read_sql_query(query, conn)
    # each row is converted into a dictionary indexed by column names
    df = df.to_dict('records')
    if event_name == 'PI stage':
        for i, e in enumerate(df):
            e['stage'] = PI_EVENTS_VALUE_MAP['PI stage'][e['value']]
    return df


def get_unique_PI_events(conn, event_name, conf):
    '''
    Returns PI events within the given time window.
    '''
    schema = 'mimiciii'
    table = 'chartevents'
    time_col = 'charttime'
    label_CV, ignored_values_CV = PI_EVENTS_CV[event_name]
    label_MV, ignored_values_MV = PI_EVENTS_MV[event_name]

    cols = f"CONCAT(tb.subject_id, '-', tb.hadm_id) as id, "
    cols += f"'{event_name}' as type, "
    cols += f"tb.{time_col} - a.admittime as t, "
    cols += f"tb.value as pi_value, "
    cols += f"regexp_replace(d.label, '\\D+', '', 'g') as pi_number, "
    cols += f"tb.icustay_id, d.dbsource, "  # extra info
    cols += 'row_number() over (partition by ' + \
        f'tb.subject_id, tb.hadm_id, tb.icustay_id, tb.value, d.label' + \
        f' order by tb.charttime )'
    table = f'{schema}.{table} tb INNER JOIN {schema}.admissions a'
    table += f' ON tb.hadm_id = a.hadm_id'
    table += f' INNER JOIN {schema}.patients p'
    table += f' ON tb.subject_id = p.subject_id'
    table += f' INNER JOIN {schema}.d_items d'
    table += f' ON tb.itemid = d.itemid'
    where = 'tb.hadm_id is NOT NULL'
    where += " AND a.diagnosis != 'NEWBORN'"
    where += f" AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob))" + \
        f">= {conf['min_age']}"
    where += f" AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob))" + \
        f"<= {conf['max_age']}"
    where += f" AND tb.{time_col} - a.admittime <= '{conf['max_hours']} hours'"
    if conf['starttime'] and conf['endtime']:
        where += f" AND tb.{time_col} >= '{conf['starttime']}'"
        where += f" AND tb.{time_col} < '{conf['endtime']}'"
    # Some events occur before the admisson date, but have the correct hadm_id.
    # Those events are ignored.
    where += f' AND tb.{time_col} >= a.admittime'
    where += f' AND tb.{time_col} <= a.dischtime'
    where += f" AND (d.label similar to '{label_CV}'" + \
        f"OR d.label SIMILAR TO '{label_MV}')"
    where += f' AND tb.value is NOT NULL'
    for value in ignored_values_CV:
        where += f" AND tb.value not similar to '{value}'"
    for value in ignored_values_MV:
        where += f" AND tb.value not similar to '{value}'"
    order_by = f'ORDER BY t ASC'
    query = f"SELECT * FROM (SELECT {cols} FROM {table} WHERE {where}) AS tmp"
    query += f" WHERE row_number=1 {order_by}"

    df = pd.read_sql_query(query, conn)
    df.drop(['row_number'], axis=1, inplace=True)
    # each row is converted into a dictionary indexed by column names
    df = df.to_dict('records')
    if event_name == 'PI stage':
        for i, e in enumerate(df):
            e['PI_stage'] = PI_EVENTS_VALUE_MAP['PI stage'][e['pi_value']]
            e['PI_state'] = conf['PI_states'][e['PI_stage']]
    for i, e in enumerate(df):
        e['PI_value_id'] = PI_EVENTS_VALUE_MAP[event_name][e['pi_value']]
    return df
