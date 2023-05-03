    '''
    n_comp = len(PI_hadms)
    get_PCA(n_comp, PI_PC_P, PI_admissions, PI_events, list(PI_etypes_P), conf, temporal=False, title='PI_PC_P_NT')
    get_PCA(n_comp, PI_PC_P, PI_admissions, PI_events, list(PI_etypes_P), conf, temporal=True, title='PI_PC_P_T')
    get_PCA(n_comp, NPI_PC_P, NPI_admissions, NPI_events, list(NPI_etypes_P), conf, temporal=False, title='NPI_PC_P_NT')
    get_PCA(n_comp, NPI_PC_P, NPI_admissions, NPI_events, list(NPI_etypes_P), conf, temporal=True, title='NPI_PC_P_T')

    get_PCA(n_comp, PI_PC_nz, PI_admissions, PI_events, list(PI_etypes), conf, temporal=False, title='PI_NT')
    get_PCA(n_comp, PI_PC_nz, PI_admissions, PI_events, list(PI_etypes), conf, temporal=True, title='PI_T')
    get_PCA(n_comp, NPI_PC_nz, NPI_admissions, NPI_events, list(NPI_etypes), conf, temporal=False, title='NPI_NT')
    get_PCA(n_comp, NPI_PC_nz, NPI_admissions, NPI_events, list(NPI_etypes), conf, temporal=True, title='NPI_T')

    admissions = PI_admissions
    for idd in NPI_admissions:
        admissions[idd] = NPI_admissions[idd]
    events = PI_events + NPI_events
    events.sort(key = lambda x: (x['type'], x['t']))
    events, PI_hadm_stage_t = process_events_PI(events, conf)
    A = build_eventgraph(admissions, events, join_rules)
    states = np.zeros(len(events))
    for e in events:
        states[e['i']] = e['pi_state']
    if not conf['vis']:
        start = time.time()
        PC_values = algebraic_PC(A, states=states)
        print("Time for PC without paths ", float(time.time() - start)/60.0, 'min' )
    PC_all, PC_nz, PC_P = process_PC_values(PC_values, conf)
    plot_PC(events, NPI_PC_nz, conf, nbins=30)
    etypes = set()
    for i, val in PC_nz.items():
        etypes.add(NPI_events[i]['type'])
    plot_PC(events, PC_P, conf, conf['PC_percentile'], nbins=10)
    etypes_P = set()
    for i, val in PC_P.items():
        etypes_P.add(events[i]['type'])
    labels = []
    for idd in admissions:
        if idd in PI_admissions:
            labels.append(1)
        else:
            labels.append(0)
    get_PCA(None, PC_P, admissions, events, list(etypes_P), conf, temporal=False, labels=labels, title='PC_P_NT')
    get_PCA(None, PC_P, admissions, events, list(etypes_P), conf, temporal=True, labels=labels, title='PC_P_T')
    get_PCA(None, PC_nz, admissions, events, list(etypes), conf, temporal=False, labels=labels, title='PC_NT')
    get_PCA(None, PC_nz, admissions, events, list(etypes), conf, temporal=True, labels=labels, title='PC_T')
    '''
