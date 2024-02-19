import networkx as nx
import pandas as pd
import numpy as np
import pprint

from teg.eventgraphs import *

def build_networkx_graph(A, events, patients, PC, conf, join_rules):
    n = len(events)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    # G.remove_nodes_from([n for n, d in G.degree if d == 0])
    if conf['node label']:
        labels = dict([(e['i'], e['type']) for e in events])
        nx.set_node_attributes(G, labels, 'label')

    # attrs = dict([(e['i'], e['type']) if 'PI' in e['type'] \
    #        else (e['i'], e['id']) for e in events])
    attrs = dict([(e['i'], e['id']) for e in events])
    nx.set_node_attributes(G, attrs, 'group')
    if PC is not None:
        if conf['scale_PC']:
            max_PC = max(PC.values())
            PC_scaled = dict([(i, 40) if 'PI' in events[i]['type'] or events[i]['type'] == 'Marker' \
                    else (i, v/max_PC * 120) for i, v in PC.items()])
        else:
            PC_scaled = dict([(i, 40) if 'PI' in events[i]['type'] or events[i]['type'] == 'Marker' \
                    else (i, v * 10000) for i, v in PC.items()])
            '''
            PC_scaled = dict([(i, 40) if 'PI' in events[i]['type'] or events[i]['type'] == 'Marker' \
                    else (i, v * 1000) for i, v in PC.items()])
            '''
        shapes = dict([(i, 'text') if PC[v] == 0.0 else (i, 'dot') for i, v in enumerate(PC)])
        shapes = dict([(i, 'diamond') if 'PI' in events[i]['type'] else (i, shape) for i, shape in shapes.items()])
        shapes = dict([(i, 'triangle') if events[i]['pi_stage'] == join_rules['max_pi_stage'] else (i, shape) for i, shape in shapes.items()])
        nx.set_node_attributes(G, PC_scaled, 'size')
        attrs = dict([(e['i'], "\n".join([str(k) + ": " + str(v)
            for k, v in e.items()]) + "\nPC: " + str(PC[e['i']]) + "\nSize: " + str(PC_scaled[e['i']])) for e in events])
    else:
        shapes = dict([(i, 'dot') for i in range(len(events))])
        shapes = dict([(i, 'diamond') if 'PI' in events[i]['type'] else (i, shape) for i, shape in shapes.items()])
        shapes = dict([(i, 'triangle') if events[i]['pi_stage'] == join_rules['max_pi_stage'] else (i, shape) for i, shape in shapes.items()])
        attrs = dict([(e['i'], "\n".join([str(k) + ": " + str(v) for k, v in e.items()])) for e in events])
    #nx.set_node_attributes(G, PC_scaled, 'value')
    nx.set_node_attributes(G, shapes, 'shape')
    nx.set_node_attributes(G, attrs, 'title')
    #attrs = nx.betweenness_centrality(G, weight='weight', normalized=True)
    if conf['edge label']:
        attrs = dict()
        for key, val in A.items():
            attrs[key] = {}
            e1 = events[key[0]]
            e2 = events[key[1]]
            s1 = patients[e1['id']]
            s2 = patients[e2['id']]
            t_max = get_t_max(e1, e2, join_rules)
            w_t, w_e, w_s, I_e, I_s = weight(s1, s2, e1, e2, join_rules, t_max )
            attrs[key] = f"Time diff: {w_t}\nEvent diff: {w_e}\nPatient diff: {w_s}"
            attrs[key] += "\nEvent Intersection:\n"
            attrs[key] += "\n".join([str(k) + ": " + str(v) for k, v in I_e.items()])
            attrs[key] += "\nPatient Intersection:\n"
            attrs[key] += "\n".join([str(k) + ": " + str(v) for k, v in I_s.items()])
            #attrs[key]['title'] = attrs[key]['value']
        nx.set_edge_attributes(G, values=attrs, name='title')
        attrs = dict([(key, val)
                     for key, val in A.items()])
        nx.set_edge_attributes(G, values=attrs, name='value')
    else:
        attrs = dict([(key, {'value': val, 'title': val})
                     for key, val in A.items()])
        nx.set_edge_attributes(G, attrs)
    # print(nx.is_directed_acyclic_graph(G))
    return G


def build_networkx_graph_example(A, events, patients, PC, conf, join_rules):
    n = len(events)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    names = {
            # ICU
            'CSRU':'Cardiac SICU',
            'MICU': 'MICU',
            'CCU': 'CICU',
            # Services
            'OMED': 'Osteopathic Medicine',
            'MED': 'Internal Medicine',
            'TSURG': 'Thoracic Surgical',
            # PI stage
            1: 'I',
            #CPTevents
            '94002': 'Pulminary-Ventilation',
            '94003': 'Pulminary-Ventilation',
            #Admissions Diagnosis
            'AORTIC DISSECTION': 'Aortic Dissection',
            'ACUTE MYELOGENOUS LEUKEMIA;CTX': 'Acute Myelogenous Leukemia',
            'ACUTE RENAL FAILURE;TELEMETRY': 'Acute Renal Failure',
            'TRACHEAL BRONCHIO MALASCIA/SDA': 'Tracheal Bronchio Malascia',
            'AORTIC STENOSIS\\AORTIC VALVE / ASCENDING AORTA REPLACEMENT /SDA': 'Aortic Stenosis',
            'MITRAL VALVE INSUFFICENCY\\MITRAL VALVE REPLACEMENT /SDA': 'Mitral Valve Insufficency'
            }
    # G.remove_nodes_from([n for n, d in G.degree if d == 0])
    if conf['node label']:
        labels = dict([(e['i'], e['type']) for e in events])
        '''
        labels = dict([(i, names[events[i]['cpt_cd']]) if label=='cptevents' else (i, label) for i, label in labels.items()])
        labels = dict([(i, names[events[i]['first_careunit']]) if label=='icustays' else (i, label) for i, label in labels.items()])
        labels = dict([(i, "Adm Dx:"+ names[events[i]['diagnosis']]) if label=='admissions' else (i, label) for i, label in labels.items()])
        labels = dict([(i, "Disch Dx:" + events[i]['diagnosis']) if label=='discharges' else (i, label) for i, label in labels.items()])
        labels = dict([(i, "Services:" + names[events[i]['curr_service']]) if label=='services' else (i, label) for i, label in labels.items()])
        labels = dict([(i, label + ":\n" + events[i]['pi_info']) if 'PI' in label else (i, label) for i, label in labels.items()])
        '''
        labels = dict([(i, label + '-' + names[events[i]['pi_value']]) if label == 'PI stage' else (i, label) for i, label in labels.items()])
        nx.set_node_attributes(G, labels, 'label')

    # attrs = dict([(e['i'], e['type']) if 'PI' in e['type'] \
    #        else (e['i'], e['id']) for e in events])
    attrs = dict([(e['i'], e['subject_id']) for e in events])
    nx.set_node_attributes(G, attrs, 'group')
    shapes = dict([(i, 'text') if PC[v] == 0.0 else (i, 'dot') for i, v in enumerate(PC)])
    shapes = dict([(i, shape) if 'PI' not in events[i]['type'] else (i, 'triangle') for i, shape in shapes.items()])
    nx.set_node_attributes(G, shapes, 'shape')
    max_PC = max(PC.values())
    PC_scaled = dict([(i, 40) if 'PI' in events[i]['type'] else (i, v/max_PC * 200) for i, v in PC.items()])
    nx.set_node_attributes(G, PC_scaled, 'size')
    #nx.set_node_attributes(G, PC_scaled, 'value')
    attrs = dict([(e['i'], "\n".join([str(k) + ": " + str(v)
        for k, v in e.items()]) + "\nPC: " + str(PC[e['i']]) + "\nSize: " + str(PC_scaled[e['i']])) for e in events])
    nx.set_node_attributes(G, attrs, 'title')
    #attrs = nx.betweenness_centrality(G, weight='weight', normalized=True)

        
    attrs = dict([(key, {'value': val, 'title': val})
                 for key, val in A.items()])
    nx.set_edge_attributes(G, attrs)
    return G
