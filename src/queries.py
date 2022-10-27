import numpy as np
import pandas as pd
import sys
import psycopg2
import pprint

import warnings

warnings.filterwarnings('ignore')

from schemas import *

# Database configuration
username = 'postgres'
password = 'postgres'
dbname = 'mimic'
schema = 'mimiciii'


def get_db_connection():
    # Connect to MIMIC-III database
    return psycopg2.connect(dbname=dbname, user=username, password=password)
    
#a patient query: [<id>, **<demography_attributes>]
#a patient dict: {<id>: **<demography_attributes>}
def get_patient_demography(conn, start_time, end_time):
    '''
    Returns patients with demography between the given time window.
    '''
    cols = f"CONCAT(patients.subject_id, '-', admissions.hadm_id) as id,"
    for table in PATIENTS:
        for col in PATIENTS[table]:
            cols += f'{table}.{col}, '
    cols += 'EXTRACT(YEAR FROM AGE(admissions.admittime, patients.dob)) as age'
    table = f'{schema}.admissions INNER JOIN {schema}.patients'
    join_cond = f'admissions.subject_id = patients.subject_id'
    #patients, admitted in the hospital within the time window
    #Some events occur before the admisson date, but have the correct hadm_id.
    #Those events are ignored.
    where_cond = f"(admissions.admittime < '{end_time}')"
    where_cond += f" AND (admissions.dischtime >= '{start_time}')"
    where_cond += f" AND admissions.diagnosis != 'NEWBORN'"
    where_cond += f" AND admissions.hadm_id is NOT NULL"
    #excluded 2616 patients with age more than 120 
    where_cond += f" AND EXTRACT(YEAR FROM AGE(admissions.admittime, patients.dob))<= 120"
    query = f'SELECT {cols} FROM {table} ON {join_cond} WHERE {where_cond}'
    df = pd.read_sql_query(query, conn)
    #print(df)
    #age = (df['admittime'] - df['dob']).dt.total_seconds()/ (60*60*24*365.25)
    #df['age'] = round(age, 2)
    #df.drop(['admittime'], axis=1, inplace=True)
    #creates a dictionary with <id> as a key.
    df = dict( [(k,v) for  k, v in zip(df.id, df.iloc[:, 1:].to_dict('records'))])
    #pprint.pprint(df)
    return df


#an event query : [<id>, <event_type>, <time>, **<event_attributes>$]
# an event dict: [{col_name: col_value, ...}]
def get_events(conn, event_name, start_time, end_time):
    '''
    Returns events within the given time window. 
    '''
    event_type, table, time_col = EVENTS[event_name] 
    t_table = 't' #time table
    all_cols = list(pd.read_sql_query(f'SELECT * FROM {schema}.{table} WHERE false', conn).astype(str))
    all_cols.sort()
    all_cols.remove('row_id')
    all_cols.remove('subject_id')
    all_cols.remove('hadm_id')
    if time_col in all_cols:
        all_cols.remove(time_col)
    else:
        t_table='a'
    cols = 't.' + ', t.'.join(all_cols)
    if event_name in EVENT_COLS_INCLUDE:
        cols = 't.'+ ', t.'.join(sorted(EVENT_COLS_INCLUDE[event_name])) 
    elif event_name in EVENT_COLS_EXCLUDE:
        cols = [col for col in all_cols if col not in EVENT_COLS_EXCLUDE[event_name]]
        cols = 't.' + ', t.'.join(cols)
    cols = f"CONCAT(t.subject_id, '-', t.hadm_id) as id, '{event_name}' as type, {t_table}.{time_col} as t, " + cols
    table = f'{schema}.{table} t INNER JOIN {schema}.admissions a'
    table += f' ON t.hadm_id = a.hadm_id'
    table += f' INNER JOIN {schema}.patients p'
    table += f' ON t.subject_id = p.subject_id'
    where_cond = 't.hadm_id is NOT NULL'
    where_cond += " AND a.diagnosis != 'NEWBORN'"
    #excluded 2616 patients with age more than 120 
    where_cond += f" AND EXTRACT(YEAR FROM AGE(a.admittime, p.dob))<= 120"
    where_cond += f" AND {t_table}.{time_col} >= '{start_time}'"
    where_cond += f" AND {t_table}.{time_col} < '{end_time}'"
    #Some events occur before the admisson date, but have the correct hadm_id.
    #Those events are ignored.
    where_cond += f' AND {t_table}.{time_col} >= a.admittime'
    where_cond += f' AND {t_table}.{time_col} <= a.dischtime'
    order_by = f'ORDER BY {t_table}.{time_col} ASC'
    match event_name:
        case 'chartevents':
            items = f'(SELECT c.itemid, count(c.itemid) as count from {schema}.chartevents c GROUP BY c.itemid) AS i'
            table += f' INNER JOIN {items} ON t.itemid=i.itemid'
            where_cond += ' AND i.count < 200000'
            where_cond += ' AND t.warning != 1'
            where_cond += ' AND t.error != 1'
            where_cond += " AND t.stopped !='D/C'd'"
        case 'noteevents':
            where_cond += ' AND t.iserror != 1'

    query = f"SELECT {cols} FROM {table} WHERE {where_cond} {order_by}"
    df = pd.read_sql_query(query, conn)
    #print(df)
    # each row is converted into a dictionary indexed by column names
    df = df.to_dict('records')
    #pprint.pprint(df)
    return df

    '''
    if event_name in EVENTS_EXCLUDE:
        for col, exclude_values in EVENTS_EXCLUDE[event_name].items():
            q_str = '(' + ','.join(str(val) for val in exclude_values) + ')'
            where_cond += f" AND t.{col}  NOT IN {q_str}"
    '''
    

if __name__ == '__main__':
    conn = get_db_connection()
    get_patient_demography(conn, '2143-01-01', '2143-02-01')
    get_events(conn, 'microbiologyevents', '2143-01-01', '2143-02-01')


