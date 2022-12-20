import numpy as np
import pandas as pd
import sys
import psycopg2
import pprint
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

from teg.schemas import *

# Database configuration
username = 'postgres'
password = 'postgres'
dbname = 'mimic'
schema = 'mimiciii'


def get_db_connection():
    # Connect to MIMIC-III database
    return psycopg2.connect(dbname=dbname, user=username, password=password)

# a patient query: [<id>, **<demography_attributes>]
# a patient dict: {<id>: **<demography_attributes>}


def get_patient_demography(conn, conf):
    '''
    Returns patients with demography between the given time window.
    '''
    cols = f"CONCAT(p.subject_id, '-', a.hadm_id) as id,"
    for table in PATIENTS:
        for col in PATIENTS[table]:
            cols += f'{table}.{col}, '
    cols += 'EXTRACT(YEAR FROM AGE(a.admittime, p.dob)) as age'
    table = f'{schema}.admissions a INNER JOIN {schema}.patients p'
    join_cond = f'a.subject_id = p.subject_id'
    # patients, admitted in the hospital within the time window
    # Some events occur before the admisson date, but have the correct hadm_id.
    # Those events are ignored.
    if conf['starttime'] and conf['endtime']:
        where = f"(a.admittime < '{conf['endtime']}')"
        where += f" AND (a.dischtime >= '{conf['starttime']}')"
    where += f" AND a.diagnosis != 'NEWBORN'"
    where += f" AND a.hadm_id is NOT NULL"
    where += f" AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob))" + \
        f">={conf['min_age']}"
    where += f" AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob))" + \
        f"<={conf['max_age']}"
    query = f'SELECT {cols} FROM {table} ON {join_cond} WHERE {where}'
    df = pd.read_sql_query(query, conn)
    # age = (df['admittime'] - df['dob']).dt.total_seconds()/ (60*60*24*365.25)
    # df['age'] = round(age, 2)
    # df.drop(['admittime'], axis=1, inplace=True)
    # creates a dictionary with <id> as a key.
    df = dict([(k, v)
              for k, v in zip(df.id, df.iloc[:, 1:].to_dict('records'))])
    return df


# an event query : [<id>, <event_type>, <time>, **<event_attributes>$]
# an event dict: [{col_name: col_value, ...}]
def get_events(conn, event_name, conf):
    '''
    Returns events within the given time window.
    '''
    event_type, table, time_col = EVENTS[event_name]
    t_table = 'tb'  # time table
    all_cols = list(
        pd.read_sql_query(
            f'SELECT * FROM {schema}.{table} WHERE false',
            conn).astype(str))
    all_cols.sort()
    all_cols.remove('row_id')
    all_cols.remove('subject_id')
    all_cols.remove('hadm_id')
    if time_col in all_cols:
        all_cols.remove(time_col)
    else:
        t_table = 'a'
    cols = 'tb.' + ', tb.'.join(all_cols)
    if event_name in EVENT_COLS_INCLUDE:
        cols = 'tb.' + ', tb.'.join(sorted(EVENT_COLS_INCLUDE[event_name]))
    elif event_name in EVENT_COLS_EXCLUDE:
        cols = [col for col in all_cols if col not in
                EVENT_COLS_EXCLUDE[event_name]]
        cols = 'tb.' + ', tb.'.join(cols)
    cols = f"CONCAT(tb.subject_id, '-', tb.hadm_id) as id," + \
        f"'{event_name}' as type" + \
        f", {t_table}.{time_col} - a.admittime as t, " + cols
    table = f'{schema}.{table} tb INNER JOIN {schema}.admissions a'
    table += f' ON tb.hadm_id = a.hadm_id'
    table += f' INNER JOIN {schema}.patients p'
    table += f' ON tb.subject_id = p.subject_id'
    where = 'tb.hadm_id is NOT NULL'
    where += " AND a.diagnosis != 'NEWBORN'"
    where += f" AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob))" + \
        f">= {conf['min_age']}"
    where += f" AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob))" + \
        f"<= {conf['max_age']}"
    where += f" AND {t_table}.{time_col} - a.admittime" + \
        f"<='{conf['max_hours']} hours'"
    if conf['starttime'] and conf['endtime']:
        where += f" AND {t_table}.{time_col} >= '{conf['starttime']}'"
        where += f" AND {t_table}.{time_col} < '{conf['endtime']}'"
    # Some events occur before the admisson date, but have the correct hadm_id.
    # Those events are ignored.
    where += f' AND {t_table}.{time_col} >= a.admittime'
    where += f' AND {t_table}.{time_col} <= a.dischtime'
    order_by = f'ORDER BY t ASC'
    match event_name:
        case 'chartevents':
            items = f'(SELECT c.itemid, count(c.itemid) as count' + \
                f'from {schema}.chartevents c GROUP BY c.itemid) AS i'
            table += f' INNER JOIN {items} ON tb.itemid=i.itemid'
            where += ' AND i.count < 200000'
            where += ' AND (tb.warning != 1 OR tb.warning IS NULL)'
            where += ' AND (tb.error != 1 OR tb.error IS NULL)'
            where += " AND (tb.stopped !='D/C''d' OR tb.stopped IS NULL)"
        case 'noteevents':
            where += ' AND (tb.iserror != 1 or tb.iserror is NULL)'
        case 'transfer':
            where += ' AND tb.curr_careunit IS NOT NULL'
            where += ' AND tb.prev_careunit IS NOT NULL'
    query = f"SELECT {cols} FROM {table} WHERE {where} {order_by}"
    df = pd.read_sql_query(query, conn)
    df['duration'] = timedelta(days=0)
    # each row is converted into a dictionary indexed by column names
    events = df.to_dict('records')
    if conf['duration'] and event_name == 'cptevents':
        events.sort(key=lambda x: (x['id'],
                                   x['cpt_cd'],
                                   x['t']))
        key = 'cpt_cd'
        prev_val = None
        prev_t = np.inf
        prev_id = None
        start = None
        t_unit = timedelta(days=1)
        del_list = []
        for i, e in enumerate(events):
            if e[key] == prev_val and \
                e['t'].days - prev_t == 1 and \
                e['id'] == prev_id:
                events[start]['duration'] += t_unit 
                prev_t = e['t'].days
                del_list.append(i)
            elif e[key] == prev_val and \
                e['t'].days - prev_t == 0 and \
                e['id'] == prev_id:
                del_list.append(i)
            else:
                start = i
                prev_val = e[key]
                prev_t = e['t'].days
                prev_id = e['id']
        del_count=0
        for i in del_list:
            del events[i - del_count]
            del_count += 1
    events.sort(key = lambda x: x['t'])
    return events

    '''
    if event_name in EVENTS_EXCLUDE:
        for col, exclude_values in EVENTS_EXCLUDE[event_name].items():
            q_str = '(' + ','.join(str(val) for val in exclude_values) + ')'
            where += f" AND t.{col}  NOT IN {q_str}"
    '''


def get_icustays(conn, conf):
    table = 'icustays'
    time_col = 'intime'
    cols = f"CONCAT(tb.subject_id, '-', tb.hadm_id) as id"
    cols += f",tb.subject_id, tb.hadm_id"
    cols += f", tb.{time_col} - a.admittime as t, tb.icustay_id"
    table = f'{schema}.{table} tb INNER JOIN {schema}.admissions a'
    table += f' ON tb.hadm_id = a.hadm_id'
    table += f' INNER JOIN {schema}.patients p'
    table += f' ON tb.subject_id = p.subject_id'
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
    order_by = f'ORDER BY t ASC'
    query = f"SELECT {cols} FROM {table} WHERE {where}"
    df = pd.read_sql_query(query, conn)
    return dict([(k, v) for k, v in zip(df.icustay_id, df.to_dict('records'))])


if __name__ == '__main__':
    conn = get_db_connection()
    get_patient_demography(conn, '2143-01-01', '2143-02-01')
    get_events(conn, 'microbiologyevents', '2143-01-01', '2143-02-01')
