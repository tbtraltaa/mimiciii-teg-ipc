import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['font.size'] = 14

from mimiciii_teg.schemas.PI_risk_factors import CHRONIC_ILLNESS
from mimiciii_teg.utils.event_utils import count_patient_events


def plot_PI_NPI_patients(PI_patients,
                         NPI_patients,
                         PI_df,
                         NPI_df,
                         conf,
                         PI_events,
                         NPI_events,
                         PI_results = None,
                         NPI_results = None,
                         P = '',
                         fname = 'Patients'):
    pi_patients = dict()
    npi_patients = dict()
    if PI_results and NPI_results:
        for idd in PI_results['patient_CENTRALITY_P']:
            pi_patients[idd] = PI_patients[idd]
        for idd in NPI_results['patient_CENTRALITY_P']:
            npi_patients[idd] = NPI_patients[idd]
        pi_df = PI_df[PI_df['id'].isin(pi_patients)] 
        npi_df = NPI_df[NPI_df['id'].isin(npi_patients)] 
    else:
        pi_patients = PI_patients
        npi_patients = NPI_patients
        pi_df = PI_df
        npi_df = NPI_df
    PI_c = 'red'
    NPI_c = 'blue'
    PI_l = 'PI'
    NPI_l = 'NPI'
    patient_dict = dict()
    chronic_dict = dict()
    unknown_ethnicities = ['unknown', 'patient declined to answer', 'unable to obtain', 'other']
    unknown_religions = ['other', 'not specified', 'unobtainable']
    for (pi_id, pi_vals), (npi_id, npi_vals) in zip(pi_patients.items(), npi_patients.items()): 
        for key, val in pi_vals.items():
            if conf['admission_type'] and 'admission_type' in key:
                continue
            elif 'age' in key:
                continue
            if key in CHRONIC_ILLNESS:
                if key not in chronic_dict:
                    chronic_dict[key] = [0, 0]
                if key in chronic_dict and val == 1:
                    chronic_dict[key][0] += 1
                continue
            if val is None:
                continue
            if key.lower() == 'ethnicity':
                if '/' in val:
                    val = val.split('/')[0].strip()
                elif '-' in val:
                    val = val.split('-')[0].strip()
                if val.lower() in unknown_ethnicities:
                    val = 'Unknown'
                elif val.lower() == 'hispanic or latino':
                    val = 'HISPANIC'
            if key.lower() == 'religion' and val.lower() in unknown_religions:
                val = 'Unknown'
            k = f"{key}-{val}"
            k = k.strip()
            if k in patient_dict:
                patient_dict[k][0] += 1
            else:
                patient_dict[k] = [1, 0]
        for key, val in npi_vals.items():
            if conf['admission_type'] and 'admission_type' in key:
                continue
            elif 'age' in key:
                continue
            if key in CHRONIC_ILLNESS:
                if key in chronic_dict and val == 1:
                    chronic_dict[key][1] += 1
                elif key not in chronic_dict and val == 1:
                    chronic_dict[key] = [0, 1]
                continue
            if val is None:
                continue
            if key.lower() == 'ethnicity':
                if '/' in val:
                    val = val.split('/')[0].strip()
                elif '-' in val:
                    val = val.split('-')[0].strip()
                if val.lower() in unknown_ethnicities:
                    val = 'Unknown'
                elif val.lower() == 'hispanic or latino':
                    val = 'HISPANIC'
            if key.lower() == 'religion' and val.lower() in unknown_religions:
                val = 'Unknown'
            k = f"{key}-{val}"
            k = k.strip()
            if k in patient_dict:
                patient_dict[k][1] += 1
            else:
                patient_dict[k] = [0, 1]
    tmp = {k: patient_dict[k] for k in patient_dict if 'religion' in k.lower() or 'ethnicity' in k.lower()}
    tmp = dict(sorted(tmp.items(), key=lambda x: x[1][0]))
    chronic_dict = dict(sorted(chronic_dict.items(), key=lambda x: x[1][0]))
    plt.style.use('default')
    plt.figure(figsize = (10, 8))
    y_pos  =  range(2, 4 * len(tmp) + 2, 4)
    PI_vals = [val[0] for val in tmp.values()]
    NPI_vals = [val[1] for val in tmp.values()]
    labels = [l.title() for l in tmp.keys()]
    plt.barh(y_pos, PI_vals, align='center', color=PI_c, label=PI_l)
    plt.yticks(y_pos, fontsize=10, labels=labels)
    y_pos  =  range(0, 4 * len(tmp), 4)
    plt.barh(y_pos, NPI_vals, align='center', color=NPI_c, label=NPI_l)
    y_pos  =  range(1, 4*len(tmp) + 1, 4)
    plt.title(f"Patient Attributes {P}")
    plt.xlabel("Frequency")
    plt.legend()
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_attrs1")
    plt.clf()
    plt.cla()

    tmp = {k: patient_dict[k] for k in patient_dict if 'religion' not in k.lower() and 'ethnicity' not in k.lower()}
    tmp = dict(sorted(tmp.items(), key=lambda x: x[1][0]))
    plt.style.use('default')
    plt.figure(figsize = (10, 8))
    y_pos  =  range(2, 4 * len(tmp) + 2, 4)
    PI_vals = [val[0] for val in tmp.values()]
    NPI_vals = [val[1] for val in tmp.values()]
    labels = [l.title() for l in tmp.keys()]
    plt.barh(y_pos, PI_vals, align='center', color=PI_c, label=PI_l)
    plt.yticks(y_pos, fontsize=10, labels=labels)
    y_pos  =  range(0, 4 * len(tmp), 4)
    plt.barh(y_pos, NPI_vals, align='center', color=NPI_c, label=NPI_l)
    y_pos  =  range(1, 4*len(tmp) + 1, 4)
    plt.title(f"Patient Attributes {P}")
    plt.xlabel("Frequency")
    plt.legend()
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_attrs2")
    plt.clf()
    plt.cla()

    if conf['include_chronic_illness']:
        plt.style.use('default')
        plt.figure(figsize = (10, 8))
        y_pos  =  range(2, 4*len(chronic_dict) + 2, 4)
        PI_vals = [val[0] for val in chronic_dict.values()]
        NPI_vals = [val[1] for val in chronic_dict.values()]
        labels = [l.title() for l in chronic_dict.keys()]
        plt.barh(y_pos, PI_vals, align='center', color=PI_c, label=PI_l)
        plt.yticks(y_pos, fontsize=10, labels=labels)
        y_pos  =  range(0, 4*len(chronic_dict), 4)
        plt.barh(y_pos, NPI_vals, align='center', color=NPI_c, label=NPI_l)
        plt.title(f"Chronic Illness {P}")
        plt.xlabel("Frequency")
        plt.legend()
        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()
        plt.savefig(f"{fname}_chronic")
        plt.clf()
        plt.cla()


    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (10, 8))
    plt.title(f"Age {P}")
    plt.hist(pi_df['age'], bins=30, rwidth=0.7, color=PI_c, label=PI_l)
    plt.hist(npi_df['age'], bins=30, rwidth=0.7, color=NPI_c, label=NPI_l)
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{fname}_age")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (10, 8))
    plt.title(f"Length Of Stay {P}")
    plt.hist(pi_df['los'], bins=30, rwidth=0.7, color=PI_c, label=PI_l)
    plt.hist(npi_df['los'], bins=30, rwidth=0.7, color=NPI_c, label=NPI_l)
    plt.xlabel("Length Of Stay In Days")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{fname}_los")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (10, 8))
    plt.title(f" OASIS {P}")
    plt.hist(pi_df['oasis'], bins=30, rwidth=0.7, color=PI_c, label=PI_l)
    plt.hist(npi_df['oasis'], bins=30, rwidth=0.7, color=NPI_c, label=NPI_l)
    plt.xlabel("OASIS")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{fname}_oasis")
    plt.clf()
    plt.cla()

    PI_patient_event_count = count_patient_events(PI_events)
    NPI_patient_event_count = count_patient_events(NPI_events)
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (10, 8))
    plt.title(f" Patient Event Count {P}")
    plt.hist(PI_patient_event_count.values(), bins=50, rwidth=0.7, color=PI_c, label=PI_l)
    plt.hist(NPI_patient_event_count.values(), bins=50, rwidth=0.7, color=NPI_c, label=NPI_l)
    plt.xlabel("Event count")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{fname}_event_count")
    plt.clf()
    plt.cla()

def plot_patients(Patients,
                         df,
                         conf,
                         events,
                         results = None,
                         P = '',
                         fname = 'Patients',
                         c = None):
    patients = dict()
    if P and results:
        for idd in results['patient_CENTRALITY_P']:
            patients[idd] = Patients[idd]
        df = df[df['id'].isin(patients)] 
        plt.style.use('default')
        plt.rcParams['font.size'] = 14
        plt.figure(figsize = (10, 8))
        plt.title(f" Centrality of Patients {P}")
        if c:
            plt.hist(results['patient_CENTRALITY_P'].values(), bins=30, rwidth=0.7, color=c)
        else:
            plt.hist(results['patient_CENTRALITY_P'].values(), bins=30, rwidth=0.7)
        plt.axvline(results['PCENTRALITY_P'][0], color='red', linestyle='dashed', linewidth=1,
                    label=f"{conf['P_patients'][0]}th Percentile")
        plt.xlabel("Centrality of Entities")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(f"{fname}_centrality_of_entities_P")
        plt.clf()
        plt.cla()
    elif results:
        patients = Patients
        plt.style.use('default')
        plt.rcParams['font.size'] = 14
        plt.figure(figsize = (10, 8))
        plt.title(f" Centrality of Patients")
        if c:
            plt.hist(results['patient_CENTRALITY_total'].values(), bins=30, rwidth=0.7, color=c)
        else:
            plt.hist(results['patient_CENTRALITY_total'].values(), bins=30, rwidth=0.7)
        plt.axvline(results['PCENTRALITY_P'][0], color='red', linestyle='dashed', linewidth=1,
                    label=f"{conf['P_patients'][0]}th Percentile")
        plt.xlabel("Centrality of Entities")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(f"{fname}_centrality_of_entities")
        plt.clf()
        plt.cla()
    else:
        patients = Patients
    patient_dict = dict()
    chronic_dict = dict()
    unknown_ethnicities = ['unknown', 'patient declined to answer', 'unable to obtain', 'other']
    unknown_religions = ['other', 'not specified', 'unobtainable']
    for id, vals in patients.items(): 
        for key, val in vals.items():
            if conf['admission_type'] and 'admission_type' in key:
                continue
            elif 'age' in key:
                continue
            if key in CHRONIC_ILLNESS:
                if key not in chronic_dict:
                    chronic_dict[key] = 0
                if key in chronic_dict and val == 1:
                    chronic_dict[key] += 1
                continue
            if val is None:
                continue
            if key.lower() == 'ethnicity':
                if '/' in val:
                    val = val.split('/')[0].strip()
                elif '-' in val:
                    val = val.split('-')[0].strip()
                if val.lower() in unknown_ethnicities:
                    val = 'Unknown'
                elif val.lower() == 'hispanic or latino':
                    val = 'HISPANIC'
            if key.lower() == 'religion' and val.lower() in unknown_religions:
                val = 'Unknown'
            k = f"{key}-{val}"
            k = k.strip()
            if k in patient_dict:
                patient_dict[k]+= 1
            else:
                patient_dict[k] = 1
    tmp = {k: patient_dict[k] for k in patient_dict if 'religion' in k.lower() or 'ethnicity' in k.lower()}
    tmp = dict(sorted(tmp.items(), key=lambda x: x[1]))
    chronic_dict = dict(sorted(chronic_dict.items(), key=lambda x: x[1]))
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (10, 8))
    y_pos  =  range(2, 4 * len(tmp) + 2, 4)
    vals = [val for val in tmp.values()]
    labels = [l.title() for l in tmp.keys()]
    plt.barh(y_pos, vals, align='center', color=c)
    plt.yticks(y_pos, fontsize=14, labels=labels)
    y_pos  =  range(0, 4 * len(tmp), 4)
    plt.title(f"Patient Attributes {P}")
    plt.xlabel("Frequency")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_attrs1")
    plt.clf()
    plt.cla()

    tmp = {k: patient_dict[k] for k in patient_dict if 'religion' not in k.lower() and 'ethnicity' not in k.lower()}
    tmp = dict(sorted(tmp.items(), key=lambda x: x[1]))
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (10, 8))
    y_pos  =  range(2, 4 * len(tmp) + 2, 4)
    vals = [val for val in tmp.values()]
    labels = [l.title() for l in tmp.keys()]
    plt.barh(y_pos, vals, align='center', color=c)
    plt.yticks(y_pos, fontsize=14, labels=labels)
    y_pos  =  range(1, 4*len(tmp) + 1, 4)
    plt.title(f"Patient Attributes {P}")
    plt.xlabel("Frequency")
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(f"{fname}_attrs2")
    plt.clf()
    plt.cla()

    if conf['include_chronic_illness']:
        plt.style.use('default')
        plt.rcParams['font.size'] = 14
        plt.figure(figsize = (10, 8))
        y_pos  =  range(2, 4*len(chronic_dict) + 2, 4)
        vals = [val for val in chronic_dict.values()]
        labels = [l.title() for l in chronic_dict.keys()]
        plt.barh(y_pos, vals, align='center', color=c)
        plt.yticks(y_pos, fontsize=14, labels=labels)
        plt.title(f"Chronic Illness {P}")
        plt.xlabel("Frequency")
        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()
        plt.savefig(f"{fname}_chronic")
        plt.clf()
        plt.cla()


    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (10, 8))
    plt.title(f"Age {P}")
    plt.hist(df['age'], bins=30, rwidth=0.7, color=c)
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.savefig(f"{fname}_age")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (10, 8))
    plt.title(f"Length Of Stay {P}")
    plt.hist(df['los'], bins=30, rwidth=0.7, color=c)
    plt.xlabel("Length Of Stay In Days")
    plt.ylabel("Frequency")
    plt.savefig(f"{fname}_los")
    plt.clf()
    plt.cla()

    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (10, 8))
    plt.title(f" OASIS {P}")
    plt.hist(df['oasis'], bins=30, rwidth=0.7, color=c)
    plt.xlabel("OASIS")
    plt.ylabel("Frequency")
    plt.savefig(f"{fname}_oasis")
    plt.clf()
    plt.cla()

    patient_event_count = count_patient_events(events)
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (10, 8))
    plt.title(f" Patient Event Count {P}")
    plt.hist(patient_event_count.values(), bins=50, rwidth=0.7, color=c)
    plt.xlabel("Event count")
    plt.ylabel("Frequency")
    plt.savefig(f"{fname}_event_count")
    plt.clf()
    plt.cla()
    
