schema = 'mimiciii'

# attributes of an event, not used for comparison
EVENT_IDs = ['id',
             'type',
             't', 
             'i',
             'j',
             'hadm_id',
             'subject_id',
             'datetime',
             'parent_type',
             'pi_stage',
             'pi_state',
             ]
IGNORE_COLS = [
    'icustay_id',
    'dbsource',
    'pi_number',
    # labs and inteventions
    'intervention-count'
    'vitals-mean',
    'vitals-count',
    'vitals-std',
    'icu-time',
    'count_Q',
    'std_Q',
    'cptdesc',
    'adm_num',
    'numeric_value',
]

LOGISTIC_EVENTS = ['Admissions', 'Transfer', 'ICU', 'Discharges', 'Services', 'Callout']

# {<table_name>: [<columns>]}
PATIENTS = {
    # p for patients
    'p': [
        'gender',
        'dob'],
    # a for admissions
    'a': [
        'admission_type',
        'insurance',
        'language',
        'religion',
        'marital_status',
        'ethnicity',
        'diagnosis']}


# {<event_name>: [<event_name>, <event_table>, <time_column>, <main attr>]}
EVENTS = {
    # Patient tracking events
    #'Admissions': ['Admissions', 'admissions', 'admittime', 'tb.admission_location'],
    #'Discharges': ['Discharges', 'admissions', 'dischtime', 'tb.discharge_location'],   
    'Admissions': ['Admissions', 'admissions', 'admittime', False],
    'Discharges': ['Discharges', 'admissions', 'dischtime', False],   
    'ICU In': ['ICU In', 'icustays', 'intime', 'tb.first_careunit'],
    'ICU Out': ['ICU Out', 'icustays', 'outtime', 'tb.last_careunit'],
    'Callout': ['Callout', 'callout', 'outcometime', 'tb.callout_service'],
    'Transfer In': ['Transfer In', 'transfers', 'intime', 'tb.curr_careunit'],
    'Transfer Out': ['Transfer Out', 'transfers', 'outtime', 'tb.curr_careunit'],

    # ICU Events
    #'Chart': [8, 'chartevents', 'charttime'],
    # charttime preferred over storetime
    # ignored endtime
    'Input CV': [ 'Input', 'inputevents_cv', 'charttime', 'd.label'],
    'Input MV': [ 'Input', 'inputevents_mv', 'starttime', 'd.label'],
    # 'datatimeevents',
    # charttime preferred over storetime
    #'Output': [10, 'outputevents', 'charttime'],
    # 'procedureevents_mv',

    # Hospital Data
    # Current Procedural Terminology
    'CPT': ['CPT', 'cptevents', 'chartdate', 'tb.cpt_cd'],
    # at the end after discharge, take discharge time from admissions table
    #'Diagnoses ICD': [12, 'diagnoses_icd', 'dischtime'],
    # at the end after discharge, take discharge time from admissions table
    #'Drgcodes': [13, 'drgcodes', 'dischtime'],
    #'Lab': [14, 'labevents', 'charttime'],
    # charttime is NULL when unknown. Hence we use chartdate.
    # There are 41772 null charttime out of 631726 which is 6.6%.
    #'Microbiology': [15, 'microbiologyevents', 'chartdate'],
    # TODO exclude notes with ISERROR=1
    # 886 ISERROR=1 out of 2083180, that is, around 0.04%
    # charttime preferred over storetime
    # 'notevents': [16, 'noteevents', 'charttime'],
    'Presc Start': ['Presc Start', 'prescriptions', 'startdate', 'tb.drug'],
    'Presc End': ['Presc End', 'prescriptions', 'enddate', 'tb.drug'],
    # at the end after discharge, take discharge time from admissions table
    #'Procedures ICD': [19, 'procedures_icd', 'dischtime'],
    'Services': ['Services', 'services', 'transfertime', 'tb.curr_service']}

#{<event_name>: [<table>, <item_col>, <value_col>, <uom_col>, <cast to dtype>, <where>]
NUMERIC_COLS = {
    #{<event_name>: [<table>, <item_col>, <value_col>, <uom_col>]
    'Input-amount':[[
        f'{schema}.inputevents_cv t INNER JOIN {schema}.d_items d on t.itemid=d.itemid',
        'd.label',
        't.amount',
        't.amountuom',],
        [f'{schema}.inputevents_mv t INNER JOIN {schema}.d_items d on t.itemid=d.itemid',
        'd.label',
        't.amount',
        't.amountuom',
        ],],
    'Presc Start-dose_val_rx':{
        'table': f'{schema}.prescriptions',
        'item_col': 'drug',
        'value_col': 'dose_val_rx',
        'uom_col': 'dose_unit_rx',
        'dtype': float, # cast varchar float
        'where': ''},
    'Presc End-dose_val_rx':{
        'table': f'{schema}.prescriptions',
        'item_col': 'drug',
        'value_col': 'dose_val_rx',
        'uom_col': 'dose_unit_rx',
        'dtype': float, # cast varchar types
        'where': ''},
    'ICU Out-los': {
        'table': f'{schema}.icustays',
        'item_col': None,
        'value_col': 'los',
        'uom_col': None,
        'dtype': None,
        'where': ''
        },
    'Transfer Out-los':{
        'table': f'{schema}.transfers',
        'item_col': None,
        'value_col': 'los',
        'uom_col': None,
        'dtype': None,
        'where': ''}
    # 'chartevents-valuenum',
    # 'labevents-valuenum',
    # 'Microbiology-dilution_value',
    # 'microbiologyevents-isolate_num' #SMAILLINT,
    # 'Output-value',
}

# {event_name : [[<include_columns>], [<exclude_columns>]]}
# ROW_ID will be excluded from all events
EVENT_COLS_EXCLUDE = {
    'Admissions': [
        'dischtime',
        'discharge_location',
        'los',
        'edregtime',
        'edouttime',
        'deathtime',
        'hospital_expire_flag', # 1 - death, 0 - survival
        'has_chartevents_data',
        # exluding columns for demography
        'insurance',
        'language',
        'religion',
        'marital_status',
        'ethnicity'],
    'Discharges': [
        'admittime',
        'admission_type',
        'admission_location',
        'edregtime',
        'edouttime',
        'deathtime',
        # discharge_location indicates if there is death/expire
        'hospital_expire_flag', # 1 - death, 0 - survival
        'has_chartevents_data',
        # exluding columns for demography
        'insurance',
        'language',
        'religion',
        'marital_status',
        'ethnicity'],
    'ICU In': [
        'icustay_id',
        'outtime',
        'last_careunit',
        'los',
        'first_wardid',
        'last_wardid'],

    'ICU Out': [
        'icustay_id',
        'intime',
        'first_careunit',
        'first_wardid',
        'last_wardid'],

    'Callout': [
        'createtime',
        'updatetime',
        'acknowledgetime',
        'firstreservationtime',
        'currentreservationtime',
        'submit_wardid',
        'curr_wardid',
        'discharge_wardid',
        'callout_wardid'],
    # kept DBSOURCE which indicates CareVue or Metavision as info.
    # It is ignored when comparing events
    'Transfer In': [
        'eventtype',
        'icustay_id',
        'prev_wardid',
        'curr_wardid'
        'outtime'],

    'Transfer Out': [
        'icustay_id',
        'eventtype',
        'prev_wardid',
        'curr_wardid'
        'intime',
        'los'],

    'Chart': [
        'warning',  # TODO Metavision - excluded if 1
        'error',  # TODO Metavision specific - excluded if 1
        'resultstatus',  # TODO CareVue - excluded
        'stopped'  # TODO CareVue specific - excluded if 'D/C'd'
    ],
    'CPT': [
        'ticket_id_seq',
        'cpt_number',
        'cpt_suffix',
        'description'],

    'Output': [
        'cgid',
        'storetime',
    ]}

EVENT_COLS_INCLUDE = {
    'Input CV': [
        'subject_id',
        'hadm_id',
        'icustay_id',
        'amount',
        'amountuom',
        # filter by positive amount
        # exclude if "D/C'd" or "Stopped"
    ],
    # stopped - D/C'd, Stopped, NotStopd, Restart, NULL
    # TODO whether to keep all
    'Input MV': [
        'subject_id',
        'hadm_id',
        'icustay_id',
        'amount',
        'amountuom',
        # Metavision specific
        # only include if statusdescription is
        # 'FinishedRunning'
    ],
    # TODO whether to keep all
    # statusdescription: Changed, Paused, FinishedRunning,
    # Stopped, Rewritten, Flushed
    'Presc Start': [
        'drug',
        'dose_val_rx',
        'dose_unit_rx',
        ],
    'Presc End': [
        'drug',
        'dose_val_rx',
        'dose_unit_rx',
        ]
}


