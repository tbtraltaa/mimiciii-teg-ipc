import numpy as np
import pandas as pd
import sys
import psycopg2
import pprint
from datetime import timedelta
import warnings


warnings.filterwarnings('ignore')

from teg.schemas import *
from teg.queries_utils import *
from teg.utils import *

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
    # compute the age interval
    left = df['age']//conf['age_interval'] * conf['age_interval'] 
    right  = left + conf['age_interval']
    df['age'] = left.astype(str) + '-' + right.astype(str)
    # creates a dictionary with <id> as a key.
    df = dict([(k, v)
              for k, v in zip(df.id, df.iloc[:, 1:].to_dict('records'))])
    return df


# an event query : [<id>, <event_type>, <time>, **<event_attributes>$]
# an event dict: [{col_name: col_value, ...}]
def get_events(conn, event_key, conf):
    '''
    Returns events within the given time window.
    '''
    event_name, table, time_col, main_attr = EVENTS[event_key]
    t_table = 'tb'  # time table
    if event_key in EVENT_COLS_INCLUDE:
        cols = 'tb.' + ', tb.'.join(sorted(EVENT_COLS_INCLUDE[event_key]))
    else:
        all_cols = list(
            pd.read_sql_query(
                f'SELECT * FROM {schema}.{table} WHERE false',
                conn).astype(str))
        all_cols.sort()
        all_cols.remove('subject_id')
        all_cols.remove('hadm_id')
        all_cols.remove('row_id')
        if time_col in all_cols:
            all_cols.remove(time_col)
        else:
            t_table = 'a'
        cols = 'tb.' + ', tb.'.join(all_cols)
        if event_name in EVENT_COLS_EXCLUDE:
            cols = [col for col in all_cols if col not in
                    EVENT_COLS_EXCLUDE[event_name]]
            cols = 'tb.' + ', tb.'.join(cols)
    ID_cols = f'''
        CONCAT(tb.subject_id, '-', tb.hadm_id) as id,
        tb.subject_id as subject_id, tb.hadm_id as hadm_id,
        CONCAT('{event_name}', '-', RTRIM(INITCAP({main_attr}), '.')) as type,
        RTRIM(INITCAP({main_attr}), '.') as item_col,
        {t_table}.{time_col} - a.admittime as t,
        {t_table}.{time_col} as datetime,
        '''
    cols = ID_cols + cols
    table = f'{schema}.{table} tb INNER JOIN {schema}.admissions a'
    table += f' ON tb.hadm_id = a.hadm_id'
    table += f' INNER JOIN {schema}.patients p'
    table += f' ON tb.subject_id = p.subject_id'
    if event_name == 'Input':
        table += f' INNER JOIN {schema}.d_items d'
        table += f' ON tb.itemid = d.itemid'
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
    where += f' AND {t_table}.{time_col} is NOT NULL'
    where += f' AND {t_table}.{time_col} >= a.admittime'
    where += f' AND {t_table}.{time_col} <= a.dischtime'
    order_by = f'ORDER BY t ASC'
    '''
    if event_name == 'chartevents':
        items = f'(SELECT c.itemid, count(c.itemid) as count' + \
            f'from {schema}.chartevents c GROUP BY c.itemid) AS i'
        table += f' INNER JOIN {items} ON tb.itemid=i.itemid'
        where += ' AND i.count < 200000'
        where += ' AND (tb.warning != 1 OR tb.warning IS NULL)'
        where += ' AND (tb.error != 1 OR tb.error IS NULL)'
        where += " AND (tb.stopped !='D/C''d' OR tb.stopped IS NULL)"
    elif event_name == 'noteevents':
        where += ' AND (tb.iserror != 1 or tb.iserror is NULL)'
    '''
    if event_name == 'Transfer In' or event_name == 'Transfer Out':
        # transfers contain admissions and discharge, we exclude them
        # we only take transfers with previous and current care unit
        where += ' AND tb.curr_careunit IS NOT NULL'
        where += ' AND tb.prev_careunit IS NOT NULL'
        where += " AND tb.eventtype='transfer'"
    if 'Presc' in event_name:
        # filter by frequency percentile range
        q = f'''SELECT RTRIM(INITCAP(drug), '.') as drug, 
            count(*) as count
            FROM {schema}.prescriptions
            WHERE drug is NOT NULL
            AND dose_val_rx is NOT NULL
            AND TRIM(dose_val_rx) != '0'
            AND TRIM(dose_val_rx) != '0.0'
            GROUP BY RTRIM(INITCAP(drug), '.')
            '''
        drug_freq = pd.read_sql_query(q, conn)
        min_count = np.nanpercentile(drug_freq['count'], conf['drug_percentile'][0])
        max_count = np.nanpercentile(drug_freq['count'], conf['drug_percentile'][1])
        presc = f"(SELECT RTRIM(INITCAP(p.drug), '.') as drug, count(*) as count" + \
            f' from {schema}.prescriptions p GROUP BY p.drug) AS q'
        table += f" INNER JOIN {presc} ON RTRIM(INITCAP(tb.drug), '.')=q.drug"
        where += f''' AND q.count >= {min_count}
            AND q.count <= {max_count}
            AND tb.drug IS NOT NULL
            AND tb.dose_val_rx is NOT NULL
            AND tb.dose_unit_rx is NOT NULL
            AND TRIM(tb.dose_val_rx) != '0'
            AND TRIM(tb.dose_val_rx) != '0.0'
            '''
    if 'Input' in event_name:
        # filter by frequency percentile range
        q = f'''SELECT RTRIM(INITCAP(d.label), '.') as input, 
            count(*) as count
            FROM {schema}.inputevents_cv i
            INNER JOIN {schema}.d_items d
            ON i.itemid = d.itemid
            WHERE i.amount is NOT NULL
            AND i.amount > 0
            AND TRIM(i.stopped) != 'D/C''d'
            AND TRIM(i.stopped) != 'Stopped'
            GROUP BY RTRIM(INITCAP(d.label), '.')
            '''
        freq_cv = pd.read_sql_query(q, conn)
        q = f'''SELECT RTRIM(INITCAP(d.label), '.') as input, 
            count(*) as count
            FROM {schema}.inputevents_mv i
            INNER JOIN {schema}.d_items d
            ON i.itemid = d.itemid
            WHERE i.amount is NOT NULL
            AND i.amount > 0
            AND TRIM(i.statusdescription) = 'FinishedRunning'
            GROUP BY RTRIM(INITCAP(d.label), '.')
            '''
        freq_mv = pd.read_sql_query(q, conn)
        freq = pd.concat([freq_cv, freq_mv])
        freq = freq.groupby('input').sum()
        min_count = np.nanpercentile(freq['count'], conf['input_percentile'][0])
        max_count = np.nanpercentile(freq['count'], conf['input_percentile'][1])
        freq = freq[freq['count']>= min_count]
        freq = freq[freq['count'] <= max_count]
        inputs = tuple([v for v in freq.index])
        where += f''' 
            AND RTRIM(INITCAP(d.label), '.') in {inputs}
            AND tb.amount is NOT NULL
            AND tb.amountuom is NOT NULL
            AND tb.amount > 0
            '''
        if event_key == 'Input MV':
            where += "AND TRIM(tb.statusdescription) = 'FinishedRunning'"
        elif event_key == 'Input CV':
            where += "AND TRIM(tb.stopped) != 'D/C''d'"
            where += "AND TRIM(tb.stopped) != 'Stopped'"

    query = f"SELECT {cols} FROM {table} WHERE {where} {order_by}"
    df = pd.read_sql_query(query, conn)
    for col in df.columns:
        uom_col = None
        if f'{event_name}-{col}' not in NUMERIC_COLS:
            continue
        if event_name == 'Input':
            args = NUMERIC_COLS[f'{event_name}-{col}']
            Q  = query_quantiles_Input(conn, conf['quantiles'], args) 
            uom_col = 'amountuom'
        else:
            args = NUMERIC_COLS[f'{event_name}-{col}']
            Q  = query_quantiles(conn, conf['quantiles'], **args) 
            if args['uom_col']:
                uom_col = args['uom_col']
        # compute percentiles for numeric values withuom
        if uom_col and conf['include_numeric']:
            df['value_test'] = df[col]
            # due to apply error in pandas, a loop is used
            vals = list()
            for idx, row in df.iterrows():
                vals.append(get_quantile_uom(row, 'item_col', col, uom_col, Q))
            df[col] = vals
        # compute percentiles for numeric values without uom
        elif conf['include_numeric']:
            df['value_test'] = df[col]
            df[col] = df[col].apply(lambda x: get_quantile(x, Q))
        # add uom to numeric values
        # drop unit of measure column
        if uom_col:
            df[col] = df[col].astype(str) + ' ' + df[uom_col]
            df.drop([uom_col], axis=1)
        # include percentiles in event type
        if conf['include_numeric']:
            df['type'] = df['type'] + ' ' + df[col]
    df.drop(['item_col'], axis=1)
    if conf['duration']:
        df['duration'] = timedelta(days=0)
    # each row is converted into a dictionary indexed by column names
    events = df.to_dict('records')
    # delete the same cpt code events of same patients happening within the same day
    # aggregate the same cpt code events of duration is true
    if 'CPT' in event_name:
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
            # the same cpt event happening next day
            if e[key] == prev_val and \
                e['t'].days - prev_t == 1 and \
                e['id'] == prev_id:
                if conf['duration']:
                    events[start]['duration'] += t_unit 
                    del_list.append(i)
                prev_t = e['t'].days
            # the same cpt event happening same day
            elif e[key] == prev_val and \
                e['t'].days - prev_t == 0 and \
                e['id'] == prev_id:
                del_list.append(i)
            else:
                start = i
                prev_val = e[key]
                prev_t = e['t'].days
                prev_id = e['id']
        for i in sorted(del_list, reverse=True):
            del events[i]
        events.sort(key = lambda x: (x['type'], x['t']))
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
    cols += f", tb.{time_col} as datetime "
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
