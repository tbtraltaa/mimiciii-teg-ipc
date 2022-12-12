import numpy as np
import pandas as pd
import sys
import psycopg2
import pprint
from datetime import timedelta
import copy
import warnings

warnings.filterwarnings('ignore')


from schemas_PI import *

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
        f" OR d.label SIMILAR TO '{label_MV}')"
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
            e['stage'] = PI_VALUE_MAP['PI stage'][e['value']]
    return df


def get_unique_PI_events(conn, event_name, conf):
    '''
    Returns PI events within the given time window.
    '''
    schema = 'mimiciii'
    table = 'chartevents'
    time_col = 'charttime'
    labels = []
    ignored_values = []
    if event_name in PI_EVENTS_CV:
        label_CV, ignored_values_CV = PI_EVENTS_CV[event_name]
        labels.append(label_CV)
        ignored_values += ignored_values_CV
    if event_name in PI_EVENTS_MV:
        label_MV, ignored_values_MV = PI_EVENTS_MV[event_name]
        labels.append(label_MV)
        ignored_values += ignored_values_MV

    cols = f"CONCAT(tb.subject_id, '-', tb.hadm_id) as id, "
    cols += f"'{event_name}' as type, "
    cols += f"tb.{time_col} - a.admittime as t, "
    cols += f"tb.value as pi_value, "
    cols += f"regexp_replace(d.label, '\\D+', '', 'g') as pi_number, "
    cols += f"tb.icustay_id, d.dbsource, "  # extra info
    cols += 'row_number() over (partition by ' + \
        f'tb.subject_id, tb.hadm_id, tb.icustay_id, tb.value, d.label' + \
        f', EXTRACT(DAY FROM tb.{time_col} - a.admittime)' + \
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
    where += f' AND tb.value is NOT NULL'
    where += f" AND (d.label similar to '{labels[0]}'"
    for label in labels[1:]:
        where += f" OR d.label SIMILAR TO '{label}'"
    where += ")"
    for value in ignored_values:
        where += f" AND tb.value not similar to '{value}'"
    order_by = f'ORDER BY t ASC'
    query = f"SELECT * FROM (SELECT {cols} FROM {table} WHERE {where}) AS tmp"
    query += f" WHERE row_number=1 {order_by}"

    df = pd.read_sql_query(query, conn)
    df.drop(['row_number'], axis=1, inplace=True)
    df['duration'] = timedelta(days=0)
    if event_name in PI_EVENTS_NUMERIC:
        df['pi_value'] = df['pi_value'].apply(lambda x: float(x.strip().split()[0]))
    elif event_name == 'PI stage':
        df['pi_value'] = df['pi_value'] \
                            .apply(lambda x: PI_STAGE_MAP[x])
        df['pi_state'] = df['pi_value'] \
                            .apply(lambda x: conf['PI_states'][x])
    elif event_name in PI_VALUE_MAP:
        df['pi_info'] = df['pi_value']
        df['pi_value'] = df['pi_value'] \
                                .apply(lambda x: PI_VALUE_MAP[event_name][x])
    else:
        df['pi_value'] = df['pi_value'].apply(lambda x: x.strip().title())
    if event_name == 'PI skin type':
        df[df['pi_value'] == 'Subq Emphysema']['pi_value'] = 'Sub Q Emphysema'

    # each row is converted into a dictionary indexed by column names
    events = df.to_dict('records')
    if conf['duration']:
        # calculating event duration for each PI
        events.sort(key=lambda x: (x['id'], x['pi_number'], x['pi_value'], x['t']))
        key = 'pi_value'
        prev_val = None
        prev_t = np.inf
        prev_id = None
        prev_num = None
        start = None
        t_unit = timedelta(days=1)
        del_list = []
        for i, e in enumerate(events):
            if e[key] == prev_val and \
                e['t'].days - prev_t == 1 and \
                e['id'] == prev_id and \
                e['pi_number'] == prev_num:
                events[start]['duration'] += t_unit 
                prev_t = e['t'].days
                del_list.append(i)
            else:
                start = i
                prev_val = e[key]
                prev_t = e['t'].days
                prev_id = e['id']
                prev_num = e['pi_number']
        del_count=0
        for i in del_list:
            del events[i - del_count]
            del_count += 1
    events.sort(key = lambda x: x['t'])
    return events
