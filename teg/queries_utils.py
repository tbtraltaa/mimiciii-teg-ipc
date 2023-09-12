import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt

from teg.schemas import *
from teg.PI_risk_factors import *

def add_chronic_illness(conn, df, conf):
    'Add chronic illness info'
    # it is slow to extract these features if the number of
    # patients are large.
    df_chronic = get_chronic_illness(conn, conf)
    df_chronic = df_chronic.set_index('hadm_id')
    q = f'''
        SELECT DISTINCT icd.icd9_code, icd.subject_id 
        FROM {schema}.diagnoses_icd icd
        INNER JOIN {schema}.admissions a
        ON icd.hadm_id = a.hadm_id
        '''
    print(df_chronic['Stroke'])
    for ill in CHRONIC_ILLNESS:
        df[ill] = 0
    for i, row in df.iterrows():
        for ill in CHRONIC_ILLNESS:
            df.loc[i, ill] = df_chronic.loc[row['hadm_id']][ill]
    return df

def get_chronic_illness(conn, conf):
    'get chronic illness info for all admissions'
    fname = f'''data/Chronic-illness-{conf['patient_history']}.h5'''
    if os.path.exists(fname):
        return pd.read_hdf(fname) 
    # it is slow to extract these features if the number of
    # patients are large.
    q = f'''SELECT hadm_id, subject_id, admittime FROM {schema}.admissions'''
    df = pd.read_sql_query(q, conn)
    df.set_index('hadm_id')
    for ill in CHRONIC_ILLNESS:
        df[ill] = 0
    q = f'''
        SELECT DISTINCT icd.icd9_code, icd.subject_id 
        FROM {schema}.diagnoses_icd icd
        INNER JOIN {schema}.admissions a
        ON icd.hadm_id = a.hadm_id
        '''
    for i, row in df.iterrows():
        for ill in CHRONIC_ILLNESS:
            icd9_codes, curr_hadm = CHRONIC_ILLNESS[ill]
            where = f'''
                    WHERE icd.subject_id = {row['subject_id']}
                    AND a.admittime > '{str(row['admittime'] - conf['patient_history'])}'
                    AND (icd.icd9_code like '{icd9_codes[0]}'
                    '''
            for code in icd9_codes[1:]:
                where += f" OR icd.icd9_code like '{code}'"
            where += ')'
            if curr_hadm:
                where += f" AND a.admittime <= '{str(row['admittime'])}'"
            else:
                 where += f" AND a.admittime < '{str(row['admittime'])}'"
            df_icd9 = pd.read_sql_query(q + where, conn)
            if df_icd9.shape[0] > 0:
                df.loc[i, ill] = 1
    if not os.path.exists(fname):
        df.to_hdf(fname, key='df', mode='w', encoding='UTF-8')
    return df

def input_filter(conn, conf, fname='output/'):
    file_name = f'data/input_count.h5'
    if os.path.exists(file_name):
        freq = pd.read_hdf(file_name) 
    else:
        # filter by frequency percentile range
        q = f'''SELECT RTRIM(INITCAP(d.label), '.') as input, 
            count(*) as count
            FROM {schema}.inputevents_cv i
            INNER JOIN {schema}.d_items d
            ON i.itemid = d.itemid
            WHERE i.amount is NOT NULL
            AND i.amountuom is NOT NULL
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
            AND i.amountuom is NOT NULL
            AND TRIM(i.statusdescription) = 'FinishedRunning'
            GROUP BY RTRIM(INITCAP(d.label), '.')
            '''
        freq_mv = pd.read_sql_query(q, conn)
        freq = pd.concat([freq_cv, freq_mv])
        freq = freq.groupby('input').sum()
        if not os.path.exists(file_name):
            freq.to_hdf(file_name, key='df', mode='w', encoding='UTF-8')
    plt.clf()
    plt.cla()
    plt.figure(figsize=(14, 8))
    plt.title(f"Input count distribution")
    plt.hist(freq['count'], bins=100, rwidth=0.7)
    plt.xlabel("Count")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Frequency")
    plt.savefig(f"{fname}Input_count_dist")
    plt.clf()
    plt.cla()
    plt.figure(figsize=(14, 8))
    plt.title(f"Input count distribution - Log")
    plt.hist(np.log10(freq['count']), bins=100, rwidth=0.7)
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    plt.savefig(f"{fname}Input_count_dist_log")
    plt.clf()
    plt.cla()
    min_count = np.nanpercentile(freq['count'], conf['input_percentile'][0])
    max_count = np.nanpercentile(freq['count'], conf['input_percentile'][1])
    freq = freq[freq['count']>= min_count]
    freq = freq[freq['count'] <= max_count]
    # escaping ' with '' for querying
    inputs = str(tuple([v.replace("'", "''") if "'" in v else v for v in freq.index]))
    inputs = inputs.replace('"', "'")
    return inputs

def prescription_filter(conn, conf, fname='output/'):
    file_name = f'data/prescription_count.h5'
    if os.path.exists(file_name):
        presc_count = pd.read_hdf(file_name) 
    else:
        # filter by frequency percentile range
        q = f'''SELECT RTRIM(INITCAP(drug), '.') as drug, 
            count(*) as count
            FROM {schema}.prescriptions
            WHERE drug is NOT NULL
            AND dose_val_rx is NOT NULL
            AND split_part(TRIM(dose_val_rx), ' ', 1) != '0'
            AND split_part(TRIM(dose_val_rx), ' ', 1) != '0.0'
            AND split_part(TRIM(dose_val_rx), '-', 1) != '0'
            AND split_part(TRIM(dose_val_rx), '-', 1) != '0.0'
            AND (split_part(TRIM(dose_val_rx), ' ', 1) similar to '\+?\d*\.?\d*'
            OR split_part(TRIM(dose_val_rx), '-', 1) similar to '\+?\d*\.?\d*')
            GROUP BY RTRIM(INITCAP(drug), '.')
            '''
        presc_count = pd.read_sql_query(q, conn)
        if not os.path.exists(file_name):
            presc_count.to_hdf(file_name, key='df', mode='w', encoding='UTF-8')
    plt.clf()
    plt.cla()
    plt.figure(figsize=(14, 8))
    plt.title(f"Prescription count distribution")
    plt.hist(presc_count['count'], bins=100, rwidth=0.7)
    plt.xlabel("Count")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Frequency")
    plt.savefig(f"{fname}Prescription_count_dist")
    plt.clf()
    plt.cla()
    plt.figure(figsize=(14, 8))
    plt.title(f"Prescription count distribution - Log")
    plt.hist(np.log10(presc_count['count']), bins=100, rwidth=0.7)
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    plt.savefig(f"{fname}Prescription_count_dist_log")
    plt.clf()
    plt.cla()
    min_count = np.nanpercentile(presc_count['count'], conf['drug_percentile'][0])
    max_count = np.nanpercentile(presc_count['count'], conf['drug_percentile'][1])
    q = f'''
        SELECT q.drug 
        FROM 
            (SELECT RTRIM(INITCAP(drug), '.') as drug, count(*) as count 
            FROM {schema}.prescriptions
            WHERE drug IS NOT NULL
            AND dose_val_rx is NOT NULL
            AND dose_unit_rx is NOT NULL
            AND split_part(TRIM(dose_val_rx), ' ', 1) != '0'
            AND split_part(TRIM(dose_val_rx), ' ', 1) != '0.0'
            AND split_part(TRIM(dose_val_rx), '-', 1) != '0'
            AND split_part(TRIM(dose_val_rx), '-', 1) != '0.0'
            AND (split_part(TRIM(dose_val_rx), ' ', 1) similar to '\+?\d*\.?\d*'
            OR split_part(TRIM(dose_val_rx), '-', 1) similar to '\+?\d*\.?\d*')
            GROUP BY drug
            ) AS q
        WHERE q.count >= {min_count} AND q.count <= {max_count}
        '''
    df = pd.read_sql_query(q, conn)
    drugs = str(tuple([v.replace("'", "''") if "'" in v else v for v in df['drug']]))
    drugs = drugs.replace('"', "'")
    return drugs

def query_quantiles(conn, quantiles, event_name, table, item_col, value_col, uom_col, dtype, where):
    if 'Presc' in event_name:
        event_name = 'Presc'
    fname = f'''data/Q{len(quantiles)}-{event_name}-{item_col}-{value_col}-{uom_col}.h5'''
    if os.path.exists(fname):
        return pd.read_hdf(fname) 
    if item_col and uom_col:
        if dtype:
            # if value is 100-500, only taking the minimum amount, 100.
            q = f'''SELECT  RTRIM(INITCAP({item_col}), '.'), {uom_col} 
                FROM {table}
                WHERE {item_col} IS NOT NULL
                AND {uom_col} IS NOT NULL
                AND {value_col} IS NOT NULL
                AND {value_col} != ''
                AND split_part(TRIM({value_col}), ' ', 1) != '0'
                AND split_part(TRIM({value_col}), ' ', 1) != '0.0'
                AND split_part(TRIM({value_col}), '-', 1) != '0'
                AND split_part(TRIM({value_col}), '-', 1) != '0.0'
                AND split_part(TRIM({value_col}), ' ', 1) similar to '\+?\d*\.?\d*'
                AND split_part(TRIM({value_col}), '-', 1) similar to '\+?\d*\.?\d*'
                {where}
                GROUP BY RTRIM(INITCAP({item_col}), '.'), {uom_col}'''
        else:
            q = f'''SELECT  RTRIM(INITCAP({item_col}), '.'), {uom_col} 
                FROM {table}
                WHERE {item_col} IS NOT NULL
                AND {uom_col} IS NOT NULL
                AND {value_col} IS NOT NULL
                AND {value_col} > 0.0
                {where}
                GROUP BY RTRIM(INITCAP({item_col}), '.'), {uom_col}'''
        index= pd.read_sql_query(q, conn)
        index = pd.MultiIndex.from_frame(index)
        df = pd.DataFrame(index=index, columns=quantiles, dtype=float)
        for item, uom in df.index:
            item_q = item
            uom_q = uom
            if "'" in item:
                item_q = item.replace("'", "''")
            if "'" in uom:
                uom_q = uom.replace("'", "''")
            if dtype:
                # numeric values
                q = f'''SELECT  split_part({value_col}, ' ', 1) as value
                    FROM {table}
                    WHERE RTRIM(INITCAP({item_col}), '.') = '{item_q}'
                    AND {uom_col} = '{uom_q}'
                    AND {value_col} IS NOT NULL
                    AND {value_col} != ''
                    AND split_part(TRIM({value_col}), ' ', 1) != '0'
                    AND split_part(TRIM({value_col}), ' ', 1) != '0.0'
                    AND split_part(TRIM({value_col}), ' ', 1) similar to '\+?\d*\.?\d*'
                    {where}'''
                item_vals = pd.read_sql_query(q, conn)
                item_vals = item_vals[item_vals['value'] != '']
                item_vals['value'] = item_vals['value'].astype(dtype)
                item_vals = item_vals[item_vals['value'] > 0]
                # numeric values such as 100-200
                q = f'''SELECT  split_part({value_col}, '-', 1) as value
                    FROM {table}
                    WHERE RTRIM(INITCAP({item_col}), '.') = '{item_q}'
                    AND {uom_col} = '{uom_q}'
                    AND {value_col} IS NOT NULL
                    AND {value_col} != ''
                    AND split_part(TRIM({value_col}), '-', 1) != '0'
                    AND split_part(TRIM({value_col}), '-', 1) != '0.0'
                    AND split_part(TRIM({value_col}), '-', 1) similar to '\+?\d*\.?\d*'
                    AND split_part(TRIM({value_col}), ' ', 1) not similar to '\+?\d*\.?\d*'
                    {where}'''
                item_vals1 = pd.read_sql_query(q, conn)
                item_vals1 = item_vals1[item_vals1['value'] != '']
                item_vals1['value'] = item_vals1['value'].astype(dtype)
                item_vals1 = item_vals1[item_vals1['value'] > 0]
                item_val =  pd.concat([item_vals, item_vals1])
            else:
                q = f'''SELECT {value_col} FROM {table}
                    WHERE RTRIM(INITCAP({item_col}), '.') = '{item_q}'
                    AND {uom_col} = '{uom_q}'
                    AND {value_col} IS NOT NULL
                    AND {value_col} > 0.0
                    {where}'''
                item_vals = pd.read_sql_query(q, conn)
            quantile_vals = item_vals.quantile(quantiles, numeric_only=True)\
                    .round(conf['quantile_round'])
            df.loc[(item, uom), :] = quantile_vals.values.reshape(-1,)
    elif not item_col and not uom_col:
        if dtype:
            q = f'''SELECT split_part({value_col}, ' ', 1) as value 
                FROM {table}
                WHERE {value_col} IS NOT NULL
                AND split_part(TRIM({value_col}), ' ', 1) similar to '\+?\d*\.?\d*'
                AND split_part(TRIM({value_col}), '-', 1) similar to '\+?\d*\.?\d*'
                {where}
                '''
            df = pd.read_sql_query(q, conn)
            df = df[df['value'] != '']
            df['value'] = df['value'].astype(dtype)
            df = df[df['value'] > 0]
            q = f'''SELECT split_part({value_col}, '-', 1) as value 
                FROM {table}
                WHERE {value_col} IS NOT NULL
                AND split_part(TRIM({value_col}), '-', 1) similar to '\+?\d*\.?\d*'
                AND split_part(TRIM({value_col}), ' ', 1) not similar to '\+?\d*\.?\d*'
                {where}
                '''
            df1 = pd.read_sql_query(q, conn)
            df1 = df[df['value'] != '']
            df1['value'] = df['value'].astype(dtype)
            df1 = df[df['value'] > 0]
            df =  pd.concat([df, df1])
        else:
            q = f'''SELECT {value_col}
                FROM {table}
                WHERE {value_col} IS NOT NULL
                AND {value_col} > 0.0
                {where}
                '''
            df = pd.read_sql_query(q, conn)
        df = df.quantile(quantiles, numeric_only=True).round(conf['quantile_round'])
    dfna = pd.isna(df)
    print('Any NaN in Qs', dfna.any())
    
    if not os.path.exists(fname):
        df.to_hdf(fname, key='df', mode='w', encoding='UTF-8')
    return df


def query_quantiles_Input(conn, quantiles, attrs):
    fname = f'''data/Q{len(quantiles)}-Input.h5'''
    if os.path.exists(fname):
        return pd.read_hdf(fname) 
    # CV
    table, item_col, value_col, uom_col = attrs[0]
    q1 = f'''SELECT  RTRIM(INITCAP({item_col}), '.') as input, {uom_col} 
        FROM {table}
        WHERE {item_col} IS NOT NULL
        AND {uom_col} IS NOT NULL
        AND {value_col} IS NOT NULL
        AND {value_col} > 0
        AND TRIM(t.stopped) != 'D/C''d'
        AND TRIM(t.stopped) != 'Stopped'
        GROUP BY RTRIM(INITCAP({item_col}), '.'), {uom_col}'''
    index1 = pd.read_sql_query(q1, conn)
    table, item_col, value_col, uom_col = attrs[1]
    # MV
    q2 = f'''SELECT  RTRIM(INITCAP({item_col}), '.') as input, {uom_col} 
        FROM {table}
        WHERE {item_col} IS NOT NULL
        AND {uom_col} IS NOT NULL
        AND {value_col} IS NOT NULL
        AND {value_col} > 0
        AND TRIM(t.statusdescription) = 'FinishedRunning'
        GROUP BY RTRIM(INITCAP({item_col}), '.'), {uom_col}'''
    index2 = pd.read_sql_query(q2, conn)
    index =  pd.concat([index1, index2])
    index = index.drop_duplicates()
    index = pd.MultiIndex.from_frame(index)
    df = pd.DataFrame(index=index, columns=quantiles, dtype=float)
    for item, uom in df.index:
        # CV
        table, item_col, value_col, uom_col = attrs[0]
        item_q = item
        uom_q = uom
        if "'" in item:
            item_q = item.replace("'", "''")
        if "'" in uom:
            uom_q = uom.replace("'", "''")
        q = f'''SELECT {value_col} FROM {table}
            WHERE RTRIM(INITCAP({item_col}), '.') = '{item_q}'
            AND {uom_col} = '{uom_q}'
            AND {value_col} IS NOT NULL
            AND {value_col} >  0
            AND TRIM(t.stopped) != 'D/C''d'
            AND TRIM(t.stopped) != 'Stopped'
            '''
        item_vals1 = pd.read_sql_query(q, conn)
        # MV
        table, item_col, value_col, uom_col = attrs[1]
        q = f'''SELECT {value_col} FROM {table}
            WHERE RTRIM(INITCAP({item_col}), '.') = '{item_q}'
            AND {uom_col} = '{uom_q}'
            AND {value_col} IS NOT NULL
            AND {value_col} > 0
            AND TRIM(t.statusdescription) = 'FinishedRunning'
            '''
        item_vals2 = pd.read_sql_query(q, conn)
        item_vals = pd.concat([item_vals1, item_vals2])
        quantile_vals = item_vals.quantile(quantiles, numeric_only=True)
        df.loc[(item, uom), :] = quantile_vals.values.reshape(-1,)
    dfna = pd.isna(df)
    print('Any NaN in Qs', dfna.any())
    if not os.path.exists(fname):
        df.to_hdf(fname, key='df', mode='w', encoding='UTF-8')
    return df
