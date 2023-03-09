import networkx as nx
import pandas as pd
import numpy as np
from pyvis.network import Network
import matplotlib.pyplot as plt
import pprint
from datetime import timedelta
import time
import warnings
warnings.filterwarnings('ignore')


from teg.schemas import *
from teg.mimic_events import *
from teg.eventgraphs import *
from teg.percolation import PC_with_target, percolation_centrality_with_target
from teg.apercolation import *

# Experiment configuration
conf = {
    'duration': False,
    'max_hours': 720,
    #'max_hours': 168,
    #'max_hours': 336,
    'min_age': 15,
    'max_age': 89,
    'age_interval': 5, # in years, for patients
    'starttime': '2143-01-14',
    #'endtime': '2143-01-21',
    'endtime': '2143-02-14',
    'min_missing_percent': 0, # for mimic extract
    'vitals_agg': 'daily',
    'vitals_X_mean': False,
    'interventions': True,
    'label': True,
    #'PI_states': {0: 0, 0.5: 0.1, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1},
    'PI_states': {0: 0, 1: 1},
    'PC_percentile': [90, 100],
    'PI_only_sql': False, # PI patients slow to query chartevents
    'PI_only': True, # Delete non PI patients after querying all events
    'PI_as_stage': True, # PI events after stage 0 are considered as stage 1 
    'unique_chartvalue_per_day_sql': False, # Take chart events with distinct values per day
    'unique_chartvalue_per_day': True,
    'scale_PC': True, # scale by max_PC
    'Top_n_PC': 20,
    'PI_vitals': False, # Use a list of vitals related to PI
    'skip_repeat': False,
    'quantiles': np.arange(0, 1.01, 0.1),
    'drug_percentile': [40, 60],
    'input_percentile': [0, 100],
    'include_numeric': True,
    'subsequent_adm': False,
}
# Event graph configuration
# t_max = [<delta days>, <delta hours>]
t_max = {
    'Admissions': timedelta(days=7, hours=0),
    'Discharges': timedelta(days=7, hours=0),
    'Icu In': timedelta(days=7, hours=0),
    'Icu_Out': timedelta(days=7, hours=0),
    'Callout': timedelta(days=7, hours=0),
    'Transfer In': timedelta(days=7, hours=0),
    'Transfer Out': timedelta(days=7, hours=0),
    'CPT': timedelta(days=7, hours=0),
    'Presc': timedelta(days=7, hours=0),
    'Services': timedelta(days=7, hours=0),
    'other': timedelta(days=7, hours=0),
    'diff_type_same_patient': timedelta(days=7, hours=0),
    'PI': timedelta(days=7, hours=0),
    'Braden': timedelta(days=7, hours=0),
    'Input': timedelta(days=7, hours=0),
}

join_rules = {
    'IDs': EVENT_IDs if not conf['duration'] else EVENT_IDs + ['duration'],
    "t_min": timedelta(days=0, hours=0, minutes=5),
    "t_max": t_max,
    'w_e_max': 0.3,  # maximum event difference
    # default event difference for different types of events
    'w_e_default': 1,
    'join_by_subject': True,
    'duration': conf['duration'],
    'duration_similarity': timedelta(days=2),
    'sequential_join': True,
    'max_pi_stage':1,
    'include_numeric': True
}


def eventgraph_mimiciii(event_list, join_rules, conf, file_name, vis=True):
    patients, events = mimic_events(event_list, join_rules, conf)
    n = len(events)
    A = build_eventgraph(patients, events, join_rules)
    states = np.zeros(n)
    for e in events:
        states[e['i']] = e['pi_state']
    #G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    #PC, V, paths = percolation_centrality_with_target(G, states=states, weight='weight')
    #start = time.time()
    #PC_vals = algebraic_PC(A, states=states)
    #print("Time for PC without paths", float(time.time() - start)/60.0)
    start = time.time()
    PC_values, V, paths = algebraic_PC_with_paths(A, states=states)
    print(float(time.time() - start)/60.0)
    PC = dict()
    PC_all = dict()
    max_PC = float(max(PC_values))
    min_PC = float(min(PC_values))
    PC_vals = []
    for i, val in enumerate(PC_values):
        v = float(val) / max_PC if conf['scale_PC'] else float(val)
        PC_vals.append(v)
        PC_all[i] = v
        if val > 0:
            PC[i] = v
    PC_sorted_events = [[k, v, events[k]] for k, v in sorted(PC.items(), key=lambda x: x[1], reverse=True)]
    pprint.pprint(PC_sorted_events[:11])
    '''
    PC_top = dict()
    num = 100 if len(PC_sorted) > 100 else len(PC_sorted)
    for i, info in enumerate(PC_sorted):
        if info[2]['type'] not in PC_top:
            PC_top[info[2]['type']] = [1, info[1]]
        else:
            PC_top[info[2]['type']][0] += 1
            PC_top[info[2]['type']][1] += info[1]
    pprint.pprint([e for e in events if e['type']=='admissions' or e['type'] =='discharges'])
    pprint.pprint(PC_top, sort_dicts=False)
    '''
    plot_PC(events, PC, conf, nbins=30)
    print("Nodes with nonzero PC", len(PC))

    PC_nz = list(PC.values())
    P_min = np.percentile(PC_nz, conf['PC_percentile'][0])
    P_max = np.percentile(PC_nz, conf['PC_percentile'][1])
    print("Nonzero PC", len(PC_nz))
    print("Min, Max PC", min_PC, max_PC)
    print("Min, Max PC scaled", min(PC_nz), max(PC_nz))
    print("Percentile", P_min, P_max)
    PC_P = dict([(i, v) for i, v in PC.items() if v >= P_min and v <= P_max])
    print("Nodes above percentile", len(PC_P))
    plot_PC(events, PC_P, conf, conf['PC_percentile'], nbins=10)

    if vis:
        G = build_networkx_graph(A, events, patients, PC_all, paths, conf)
        file_name += "_Q" + str(len(conf['quantiles']))
        visualize_SP_tree(G, V, paths, file_name+"SP")
        visualize_SP_tree(G, list(PC_P.keys()), paths, file_name+"SP_percentile")
        attrs = dict([(e['i'], e['type']) for e in events])
        if conf['max_hours'] <= 168 or conf['PI_patients']:
            visualize_graph(G, V, paths, file_name+"all")
        nx.set_node_attributes(G, attrs, 'group')
        visualize_vertices(G, list(PC_P.keys()), file_name+"V_percentile")


def build_networkx_graph(A, events, patients, PC, paths, conf):
    n = len(events)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    # G.remove_nodes_from([n for n, d in G.degree if d == 0])
    if conf['label']:
        labels = dict([(e['i'], e['type']) for e in events])
        nx.set_node_attributes(G, labels, 'label')

    # attrs = dict([(e['i'], e['type']) if 'PI' in e['type'] \
    #        else (e['i'], e['id']) for e in events])
    attrs = dict([(e['i'], e['id']) for e in events])
    nx.set_node_attributes(G, attrs, 'group')
    shapes = dict([(i, 'text') if PC[v] == 0.0 else (i, 'dot') for i, v in enumerate(PC)])
    shapes = dict([(i, 'diamond') if 'PI' in events[i]['type'] else (i, shape) for i, shape in shapes.items()])
    shapes = dict([(i, 'triangle') if events[i]['pi_stage'] == join_rules['max_pi_stage'] else (i, shape) for i, shape in shapes.items()])
    nx.set_node_attributes(G, shapes, 'shape')
    if not conf['scale_PC']:
        max_PC = max(PC.values())
        PC_scaled = dict([(i, 40) if 'PI' in events[i]['type'] else (i, v/max_PC * 120) for i, v in PC.items()])
    else:
        PC_scaled = dict([(i, 40) if 'PI' in events[i]['type'] else (i, v * 120) for i, v in PC.items()])
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

def visualize_SP_tree(G, V, paths, file_name):
    PC_edges = list()
    for v in V:
        for path in paths[v]:
            i = path[0]
            for j in path[1:]:
                PC_edges.append((i,j))
                i = j
    g = Network(
        directed=True,
        height=1000,
        width='90%',
        neighborhood_highlight=True,
        select_menu=True)
    g.repulsion()
    g.from_nx(G.edge_subgraph(PC_edges), show_edge_weights=False)
    # g.barnes_hut()
    g.toggle_physics(True)
    g.show_buttons()

    print(nx.is_directed_acyclic_graph(G))
    # g.show("mimic.html")
    g.save_graph(file_name + '.html')

def visualize_vertices(G, V, file_name):
    G.remove_edges_from(list(G.edges()))
    g = Network(
        directed=True,
        height=1000,
        width='90%',
        neighborhood_highlight=True,
        select_menu=True)
    g.repulsion()
    g.from_nx(G.subgraph(V), show_edge_weights=False)
    # g.barnes_hut()
    g.toggle_physics(True)
    g.show_buttons()
    print(nx.is_directed_acyclic_graph(G))
    # g.show("mimic.html")
    g.save_graph(file_name + '.html')


def visualize_graph(G, V, paths, file_name):
    for v in V:
        for path in paths[v]:
            i = path[0]
            for j in path[1:]:
                G[i][j]['color']='black'
                i = j
    g = Network(
        directed=True,
        height=1000,
        width='90%',
        neighborhood_highlight=True,
        select_menu=True)
    g.repulsion()
    g.from_nx(G, show_edge_weights=False)
    # g.barnes_hut()
    g.toggle_physics(True)
    '''
    options = {
                    "font": {
                      "size": 35
                    },
                    "selfReference": {
                      "angle": 0.7853981633974483
                    },
                  "physics": {
                    "minVelocity": 0.75,
                    "solver": "repulsion"
                  }
                }
    g.set_options(options)
    '''
    g.show_buttons()
    print(nx.is_directed_acyclic_graph(G))
    # g.show("mimic.html")
    g.save_graph(file_name + '.html')
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.15, seed=4572321)
    patient_i = dict([(id_, i) for i, id_ in enumerate(patients.keys())])
    node_color = [patient_i[e['id']] for e in events]
    node_size = [v*20000 for v in PC.values()]
    nx.draw_networkx(
        G,
        pos=pos,
        arrows=True,
        with_labels=True,
        node_color=node_color,
        node_size=node_size,
        edge_color="gainsboro",
        alpha=0.4,
    )

    # Title/legend
    font = {"color": "k", "fontweight": "bold", "fontsize": 20}
    ax.set_title("Percolation Centrality", font)
    # Change font color for legend
    font["color"] = "r"
    ax.text(
        0.80,
        0.06,
        "node size = percolation centrality",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )

    # Resize figure for label readibility
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")
    plt.show()
    '''


def build_networkx_graph_example(A, events, patients, PC, paths, conf):
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
    if conf['label']:
        labels = dict([(e['i'], e['type']) for e in events])
        labels = dict([(i, names[events[i]['cpt_cd']]) if label=='cptevents' else (i, label) for i, label in labels.items()])
        labels = dict([(i, names[events[i]['first_careunit']]) if label=='icustays' else (i, label) for i, label in labels.items()])
        labels = dict([(i, "Adm Dx:"+ names[events[i]['diagnosis']]) if label=='admissions' else (i, label) for i, label in labels.items()])
        labels = dict([(i, "Disch Dx:" + events[i]['diagnosis']) if label=='discharges' else (i, label) for i, label in labels.items()])
        labels = dict([(i, "Services:" + names[events[i]['curr_service']]) if label=='services' else (i, label) for i, label in labels.items()])
        labels = dict([(i, label + ":\n" + events[i]['pi_info']) if 'PI' in label else (i, label) for i, label in labels.items()])
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


def plot_PC(events, PC, conf, percentile='', nbins=30):
    PC_t = dict([(events[i]['type'], v) for i, v in PC.items()])
    #df = pd.DataFrame({'type': list(PC_type.keys()), 'val': list(PC_type.values())})
    #df.pivot(columns="type", values="val").plot.hist(bins=nbins)
    plt.figure(figsize=(10, 6))
    plt.title("PC value distribution " + str(percentile))
    plt.hist(list(PC.values()), bins=30, rwidth=0.7)
    plt.xlabel("Nonzero PC values")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(10, 6))
    #df['val_log10'] = np.log10(list(PC.values()))
    #df.pivot(columns="type", values="val_log10").plot.hist(bins=nbins)
    plt.title("PC value distribution after Log transformation " + str(percentile))
    plt.hist(np.log10(list(PC.values())), bins=20, rwidth=0.7)
    plt.xlabel("Nonzero PC values")
    #plt.yscale("log")
    plt.ylabel("Frequency")
    plt.show()

    '''
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        PC_tt = dict()
        for t, v in PC_t.items():
            if t not in PC_t:
                PC_tt[t] = [v]
            else:
                PC_tt[t].append(v)
        ys = []
        cmap = plt.get_cmap('inferno')
        for i, t in enumerate(PC_tt):
            hist, bins = np.histogram(PC_tt[t], bins=nbins)
            xs = (bins[:-1] + bins[1:])/2
            ys.append(i * 10)
            ax.bar(xs, hist, zs=i*10, zdir='y', color=cmap(i), alpha=0.8)
        ax.set_xlabel('PC Value')
        ax.set_ylabel('Event types')
        ax.set_zlabel('Frequency')
        ax.set_yticks(ys, labels=list(PC_tt.keys()), fontsize=14)
        plt.show()
    '''

    plt.figure(figsize=(10, 6))
    PC_t_sorted = dict(sorted(PC_t.items(), key=lambda x: x[1], reverse=True))
    top_n = conf['Top_n_PC'] if len(PC_t) > conf['Top_n_PC'] else len(PC_t)
    vals = []
    labels = []
    for key in list(PC_t_sorted.keys())[:top_n]:
        labels = [key] + labels
        vals = [PC_t_sorted[key]] + vals
    y_pos  =  range(0, 2*len(PC_t), 2)[:top_n]
    plt.barh(y_pos, vals, align='center')
    plt.yticks(y_pos, labels=labels, fontsize=14)
    plt.title(f"Top {top_n} PC Events" + str(percentile))
    plt.xlabel("PC Value")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    PC_n = len(PC)
    PC_freq = dict()
    PC_sum = dict()
    PC_total = 0
    for i, val in PC.items():
        etype = events[i]['type']
        PC_total += val
        if etype not in PC_freq:
            PC_freq[etype] = 1
            PC_sum[etype] = val
        else:
            PC_freq[etype] += 1
            PC_sum[etype] += val
    PC_freq = dict(sorted(PC_freq.items(), key=lambda x: x[1]))
    y_pos  =  range(0, 2*len(PC_freq), 2)
    plt.barh(y_pos, PC_freq.values(), align='center')
    plt.yticks(y_pos, labels=list(PC_freq.keys()))
    plt.title("PC Event Type Distribution " + str(percentile))
    plt.xlabel("Frequency")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 6))
    PC_w = dict()
    PC_p = dict()
    for k, f in PC_freq.items():
        PC_w[k] = PC_sum[k]* f/PC_n
        PC_p[k] = PC_sum[k]/PC_total
    PC_w = dict(sorted(PC_w.items(), key=lambda x: x[1]))
    PC_p = dict(sorted(PC_p.items(), key=lambda x: x[1]))
    #plt.bar(PC_w.keys(), PC_w.values(), width=0.5, align='center')
    y_pos  =  range(0, 2*len(PC_w), 2)
    plt.barh(y_pos, list(PC_w.values()), align='center')
    plt.yticks(y_pos, labels=list(PC_w.keys()))
    plt.title("Weighted Average PC " + str(percentile))
    plt.xlabel("Weighted Average PC")
    plt.xscale("log")
    #plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    #plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.barh(y_pos, list(PC_p.values()), align='center')
    plt.yticks(y_pos, labels=list(PC_p.keys()))
    plt.title("Portion of total PC" + str(percentile))
    plt.xlabel("Portion of total PC")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()

def analyze_paths(events, PC, V, paths):
    for i in V:
        for path in paths[i]:
            u = path[0]
            for v in path[1:]:
                if events[u]['type'] == events[v]['type']:
                    pass
                else:
                    pass

if __name__ == "__main__":
    fname_keys = [
        'max_hours',
        'starttime',
        'endtime',
        'PC_percentile',
        'drug_percentile',
        'input_percentile',
        'skip_repeat',
        'PI_patients']
    fname_LF = 'output/DTEG'
    fname_LF += '_' + '_'.join([k + '-' + str(v)
                               for k, v in conf.items() if k in fname_keys])
    fname_LF += '_' + '_'.join([k + '-' + str(v)
                                    for k, v in join_rules.items()
                                    if k in fname_keys])

    eventgraph_mimiciii(EVENTS, join_rules, conf, fname_LF)
