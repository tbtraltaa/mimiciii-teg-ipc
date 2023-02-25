import pandas as pd

def query_quantiles(conn, quantiles, table, item_col, value_col, uom_col, dtype, where):
    if item_col and uom_col:
        if dtype:
            q = f'''SELECT  RTRIM(INITCAP({item_col}), '.'), {uom_col} 
                FROM {table}
                WHERE {item_col} IS NOT NULL
                AND {uom_col} IS NOT NULL
                AND {value_col} IS NOT NULL
                {where}
                GROUP BY RTRIM(INITCAP({item_col}), '.')), {uom_col}'''
        else:
            q = f'''SELECT  RTRIM(INITCAP({item_col}), '.'), {uom_col} 
                FROM {table}
                WHERE {item_col} IS NOT NULL
                AND {uom_col} IS NOT NULL
                AND {value_col} IS NOT NULL
                AND {value_col} > 0.0
                {where}
                GROUP BY RTRIM(INITCAP({item_col}), '.')), {uom_col}'''
        index= pd.read_sql_query(q, conn)
        df = pd.DataFrame(index=index, columns=quantiles, dtype=float)
        for item, uom in df.index:
            if dtype:
                q = f'''SELECT q.value from (SELECT  cast(split_part({value_col}, ' ', 1) as {dtype})
                    as value
                    FROM {table}
                    WHERE RTRIM(INITCAP({item_col}), '.') = '{item}'
                    AND {uom_col} = '{uom}'
                    AND {value_col} IS NOT NULL
                    {where}) as q
                    where q.value > 0'''
            else:
                q = f'''SELECT {value_col} FROM {table}
                    WHERE RTRIM(INITCAP({item_col}), '.') = '{item}'
                    AND {uom_col} = '{uom}'
                    AND {value_col} IS NOT NULL
                    AND {value_col} > 0.0
                    {where}'''
            item_vals = pd.read_sql_query(q, conn)
            quantile_vals = item_vals.quantile(quantiles, numeric_only=True)
            df.loc[(item, uom), :] = quantile_vals
    elif not item_col and not uom_col:
        if dtype:
            q = f'''SELECT q.value FROM (SELECT  cast(split_part({value_col}, ' ', 1) as {dtype}) 
                as value
                FROM {table}
                WHERE {value_col} IS NOT NULL
                {where}) as q
                where q.value > 0
                '''
        else:
            q = f'''SELECT {value_col}
                FROM {table}
                WHERE {value_col} IS NOT NULL
                AND {value_col} > 0.0
                {where}
                '''
        df = pd.read_sql_query(q, conn)
        df = df.quantile(quantiles, numeric_only=True)
    return df


def query_quantiles_Input(conn, quantiles, attrs):
    # CV
    table, item_col, value_col, uom_col = attrs[0]
    q1 = f'''SELECT  RTRIM(INITCAP({item_col}), '.'), {uom_col} 
        FROM {table}
        WHERE {item_col} IS NOT NULL
        AND {uom_col} IS NOT NULL
        AND {value_col} IS NOT NULL
        AND {value_col} > 0
        GROUP BY RTRIM(INITCAP({item_col}), '.')), {uom_col}'''
    index1 = pd.read_sql_query(q, conn)
    table, item_col, value_col, uom_col = attrs[1]
    # MV
    q2 = f'''SELECT  RTRIM(INITCAP({item_col}), '.'), {uom_col} 
        FROM {table}
        WHERE {item_col} IS NOT NULL
        AND {uom_col} IS NOT NULL
        AND {value_col} IS NOT NULL
        AND {value_col} > 0
        GROUP BY RTRIM(INITCAP({item_col}), '.')), {uom_col}'''
    index2 = pd.read_sql_query(q2, conn)
    index =  pd.concat(index1, index2)
    index.drop_duplicates()
    df = pd.DataFrame(index=index, columns=quantiles, dtype=float)
    for item, uom in df.index:
        # CV
        table, item_col, value_col, uom_col = attrs[0]
        q1 = f'''SELECT {value_col} FROM {table}
            WHERE RTRIM(INITCAP({item_col}), '.') = '{item}'
            AND {uom_col} = '{uom}'
            AND {value_col} IS NOT NULL
            AND {value_col} >  0'''
        item_vals1 = pd.read_sql_query(q1, conn)
        # MV
        table, item_col, value_col, uom_col = attrs[1]
        q1 = f'''SELECT {value_col} FROM {table}
            WHERE RTRIM(INITCAP({item_col}), '.') = '{item}'
            AND {uom_col} = '{uom}'
            AND {value_col} IS NOT NULL
            AND {value_col} > 0'''
        item_vals2 = pd.read_sql_query(q1, conn)
        item_vals = pd.concat(item_val1, item_val2)
        quantile_vals = item_vals.quantile(item_vals, numeric_only=True)
        df.loc[(item, uom), :] = quantile_vals
    return df
