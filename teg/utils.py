import pandas as pd

def get_quantile(val, Q):
    if type(val) == str:
        if '-' in val:
            val = float(val.split('-')[0])
        else:
            val = float(val)
    prev_q = 0
    prev_idx = 0
    for idx in Q.index:
        if Q.loc[prev_q][0] <= val and val < Q.loc[idx][0]:
            break
        if Q.loc[prev_idx][0] != Q.loc[idx][0]:
            prev_q = idx
        prev_idx = idx
    val1 = round(Q.loc[prev_q][0], 1)
    val2 = round(Q.loc[idx][0], 1)
    prev_q = round(prev_q*100)
    q = round(idx*100)
    #if prev_q == 0:
    #    val1 = ''
    if q == 100:
        val2 = ''
    return f'{prev_q}-{q}P, {val1}-{val2}'

def get_quantile_mimic_extract(val, Q):
    if type(val) == str:
        val = float(val)
    prev_q = 0
    prev_idx = 0
    for idx in Q.index:
        if Q.loc[prev_q] <= val and val < Q.loc[idx]:
            break
        if Q.loc[prev_idx] != Q.loc[idx]:
            prev_q = idx
        prev_idx = idx
    val1 = round(Q.loc[prev_q], 1)
    val2 = round(Q.loc[idx], 1)
    prev_q = round(prev_q*100)
    q = round(idx*100)
    #if prev_q == 0:
    #    val1 = ''
    if q == 100:
        val2 = ''
    return f'{prev_q}-{q}P, {val1}-{val2}'

def get_quantile_uom(row, item_col, val_col, uom_col, Q):
    val = row[val_col]
    uom = row[uom_col]
    item = row[item_col]
    Q_item = pd.DataFrame(Q.loc[(item, uom), :].values, index=Q.columns)
    val_P = get_quantile(val, Q_item)
    return val_P
