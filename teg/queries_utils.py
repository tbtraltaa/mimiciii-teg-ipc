import pandas as pd

def query_quantiles(conn, quantiles, table, item_col, value_col, uom_col, dtype, where):
    if item_col and uom_col:
        if dtype:
            q = f'''SELECT  RTRIM(INITCAP({item_col}), '.'), {uom_col} 
                FROM {table}
                WHERE {item_col} IS NOT NULL
                AND {uom_col} IS NOT NULL
                AND {value_col} IS NOT NULL
                AND split_part({value_col}, ' ', 1) != '0'
                AND split_part({value_col}, ' ', 1) != '0.0'
                AND split_part({value_col}, ' ', 1) != '-0.0'
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
                q = f'''SELECT  split_part({value_col}, ' ', 1) as value
                    FROM {table}
                    WHERE RTRIM(INITCAP({item_col}), '.') = '{item_q}'
                    AND {uom_col} = '{uom_q}'
                    AND {value_col} IS NOT NULL
                    {where}'''
                item_vals = pd.read_sql_query(q, conn)
                item_vals['value'] = item_vals.astype(dtype)
                item_vals = item_vals[item_vals['value'] > 0]
            else:
                q = f'''SELECT {value_col} FROM {table}
                    WHERE RTRIM(INITCAP({item_col}), '.') = '{item_q}'
                    AND {uom_col} = '{uom_q}'
                    AND {value_col} IS NOT NULL
                    AND {value_col} > 0.0
                    {where}'''
                item_vals = pd.read_sql_query(q, conn)
            quantile_vals = item_vals.quantile(quantiles, numeric_only=True)
            df.loc[(item, uom), :] = quantile_vals
    elif not item_col and not uom_col:
        if dtype:
            q = f'''SELECT split_part({value_col}, ' ', 1) as value 
                FROM {table}
                WHERE {value_col} IS NOT NULL
                {where}
                '''
            df = pd.read_sql_query(q, conn)
            df['value'] = df['value'].astype(dtype)
            df = df[df['value'] > 0]
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
    index.drop_duplicates()
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
        df.loc[(item, uom), :] = quantile_vals
    return df
