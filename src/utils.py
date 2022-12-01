def numeric_cols(events_list):
    float_cols = set()
    int_cols = set()
    for e in events_list:
        for k, v in e.items():
            if isinstance(v, (float)):
                float_cols.add(types[e['type']] + '-' + k)
            if isinstance(v, (int)):
                int_cols.add(types[e['type']] + '-' + k)
    pprint.pprint(float_cols)
    pprint.pprint(int_cols)
