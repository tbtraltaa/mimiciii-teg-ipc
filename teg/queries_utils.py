import pandas as pd

def query_items_quantiles(conn, table, item_col, value_col, uom_col, quantiles):
        q = f'''SELECT  RTRIM(INITCAP({item_col}), '.'), {uom_col} 
            FROM {table}
            WHERE {item_col} IS NOT NULL
            AND {uom_col} IS NOT NULL
            AND {value_col} IS NOT NULL
            AND cast({value_col} as double precision) != 0
            GROUP BY RTRIM(INITCAP({item_col}), '.')), {uom_col}'''
        index= pd.read_sql_query(q, conn)
        df = pd.DataFrame(index=index, columns=quantiles, dtype=float)

        for item, uom in df.index:
            q = f'''SELECT  cast({value_col} as double precision) FROM {table}
                WHERE RTRIM(INITCAP({item_col}), '.') = '{item}'
                AND {uom_col} = '{uom}'
                AND {value_col} IS NOT NULL
                AND cast({value_col} as double precision) != 0'''
            item_vals = pd.read_sql_query(q, conn)
            quantile_vals = item_vals.quantile(quantiles, numeric_only=True)
            df.loc[(item, uom), :] = quantile_vals
        return df

def query_item_quantiles(conn, table, value_col, where, quantiles):
        q = f'''SELECT  RTRIM(INITCAP({item_col}), '.') 
            FROM {table}
            WHERE {value_col} IS NOT NULL
            AND cast({value_col} as double precision) != 0
            '''
        index= pd.read_sql_query(q, conn)
        df = pd.DataFrame(index=index, columns=quantiles, dtype=float)

        for item, uom in df.index:
            q = f'''SELECT  cast({value_col} as double precision) FROM {table}
                WHERE RTRIM(INITCAP({item_col}), '.') = '{item}'
                AND {uom_col} = '{uom}'
                AND {value_col} IS NOT NULL
                AND cast({value_col} as double precision) != 0'''
            item_vals = pd.read_sql_query(q, conn)
            quantile_vals = item_vals.quantile(quantiles, numeric_only=True)
            df.loc[(item, uom), :] = quantile_vals
        return df

def get_quantile(val, Q):
    prev_q = 0
    prev_idx = 0
    for idx in Q.index:
        if Q.loc[prev_q] <= val and val < Q.loc[idx]:
            break
        if Q.loc[prev_idx] != Q.loc[idx]:
            prev_q = idx
        prev_idx = idx
    return round(prev_q*100), round(idx*100), round(Q.loc[prev_q]), round(Q.loc[idx])

'''
def get_quantile(val, Q):
    q = None
    for q_val in Q.index:
        if val >= Q.loc[q_val]:
            q = q_val
    return q
'''
