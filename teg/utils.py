import pandas as pd


def dict_intersection_and_differences(D1, D2):
    '''
    Intersection and differences of two dictionaries
    based on dictionary keys
    '''
    A = set(D1.keys())
    B = set(D2.keys())
    I = A & B
    A_minus_B  = A - I
    B_minus_A = B - I
    A_minus_B_dict = {}
    B_minus_A_dict = {}
    I_dict = {}
    for key in I:
        I_dict[key] = [D1[key], D2[key]]
    for key in A_minus_B:
        A_minus_B_dict[key] = D1[key]
    for key in B_minus_A:
        B_minus_A_dict[key] = D2[key]
    return A_minus_B_dict, B_minus_A_dict, I_dict

def list_intersection_and_differences(L1, L2):
    '''
    List intersection and differences
    '''
    A = set(L1)
    B = set(L2)
    I = A & B
    A_minus_B  = A - I
    B_minus_A = B - I
    return list(A_minus_B), list(B_minus_A), list(I)

def get_quantile(val, Q, conf):
    # for '0.25 to 1', 0.25 is taken
    if type(val) == str:
        if '-' in val:
            val = float(val.split('-')[0])
        elif ' ' in val:
            val = float(val.split(' ')[0])
        else:
            val = float(val)
    prev_q = 0
    prev_idx = 0
    for idx in Q.index[1:]:
        val1 = round(Q.loc[prev_q][0], conf['quantile_round'])
        val2 = round(Q.loc[idx][0], conf['quantile_round'])
        prev_val = round(Q.loc[prev_idx][0], conf['quantile_round'])
        if val1 <= val and val < val2:
            break
        elif prev_val != val2:
            prev_q = idx
        prev_idx = idx
    prev_q = round(prev_q*100)
    q = round(idx*100)
    #if prev_q == 0:
    #    val1 = ''
    if q == 100:
        val2 = ''
    return f'{prev_q}-{q}P, {val1}-{val2}'


def get_quantile_interval(val, Q, conf):
    # for '0.25 to 1', 0.25 is taken
    if type(val) == str:
        if '-' in val:
            val = float(val.split('-')[0])
        elif ' ' in val:
            val = float(val.split(' ')[0])
        else:
            val = float(val)
    prev_q = 0
    prev_idx = 0
    for idx in Q.index[1:]:
        val1 = round(Q.loc[prev_q][0], conf['quantile_round'])
        val2 = round(Q.loc[idx][0], conf['quantile_round'])
        prev_val = round(Q.loc[prev_idx][0], conf['quantile_round'])
        if val1 <= val and val < val2:
            break
        elif prev_val != val2:
            prev_q = idx
        prev_idx = idx
    prev_q = round(prev_q*100)
    q = round(idx*100)
    #if prev_q == 0:
    #    val1 = ''
    if q == 100:
        val2 = ''
    return [prev_q, q]


def get_quantile_mimic_extract(val, Q, conf):
    if type(val) == str:
        val = float(val)
    prev_q = 0
    prev_idx = 0
    for idx in Q.index[1:]:
        val1 = round(Q.loc[prev_q], conf['quantile_round'])
        val2 = round(Q.loc[idx], conf['quantile_round'])
        prev_val = round(Q.loc[prev_idx], conf['quantile_round'])
        if val1 <= val and val < val2:
            break
        elif prev_val != val2:
            prev_q = idx 
        prev_idx = idx
    prev_q = round(prev_q*100)
    q = round(idx*100)
    #if prev_q == 0:
    #    val1 = ''
    if q == 100:
        val2 = ''
    return f'{prev_q}-{q}P, {val1}-{val2}', [prev_q, q]


def get_quantile_uom(row, item_col, val_col, uom_col, Q, conf):
    val = row[val_col]
    uom = row[uom_col]
    item = row[item_col]
    val_P = 'None'
    if (item, uom) in Q.index:
        Q_item = pd.DataFrame(Q.loc[(item, uom), :].values, index=Q.columns)
        val_P = get_quantile(val, Q_item, conf)
        I = get_quantile_interval(val, Q_item, conf)
    else:
        print('Quantile Missing ', item, uom)
    return val_P, I 
