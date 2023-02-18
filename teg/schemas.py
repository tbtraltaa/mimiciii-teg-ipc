# {<table_name>: [<columns>]}

# attributes of an event, not used for comparison
EVENT_IDs = ['id', 'type', 't', 'i', 'hadm_id', 'subject_id', 'time', 'adm_num']

PATIENTS = {
    # p for patients
    'p': [
        'gender',
        'dob'],
    # a for admissions
    'a': [
        'insurance',
        'language',
        'religion',
        'marital_status',
        'ethnicity',
        'diagnosis']}


# {<event_name>: [<event_number>, <event_table>, <time_column>, <main attr>]}
EVENTS = {
    # Patient tracking events
    'Admissions': [1, 'admissions', 'admittime', 'admission_location'],
    'Discharges': [2, 'admissions', 'dischtime', 'discharge_location'],
    'ICU In': [3, 'icustays', 'intime', 'first_careunit'],
    'ICU Out': [3, 'icustays', 'outtime', 'last_careunit'],
    'Callout': [5, 'callout', 'outcometime', 'callout_service'],
    'Transfer In': [6, 'transfers', 'intime', 'curr_careunit'],
    'Transfer Out': [7, 'transfers', 'outtime', 'curr_careunit'],

    # ICU Events
    #'Chart': [8, 'chartevents', 'charttime'],
    # charttime preferred over storetime
    #'Input CV': [9, 'inputevents_cv', 'charttime', None],
    # ignored endtime
    #'Input MV': [9, 'inputevents_mv', 'starttime', None],
    # 'datatimeevents',
    # charttime preferred over storetime
    #'Output': [10, 'outputevents', 'charttime'],
    # 'procedureevents_mv',

    # Hospital Data
    # Current Procedural Terminology
    'CPT': [11, 'cptevents', 'chartdate', 'cpt_cd'],
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
    'Presc Start': [17, 'prescriptions', 'startdate', 'drug'],
    'Presc End': [18, 'prescriptions', 'enddate', 'drug'],
    # at the end after discharge, take discharge time from admissions table
    #'Procedures ICD': [19, 'procedures_icd', 'dischtime'],
    'Services': [20, 'services', 'transfertime', 'curr_service']}

#{<event_name>: [<table>, <item_col>, <value_col>, <uom_col>, <where>]
NUMERIC_COLS = {
    'Input-amount':
        [
            [
                'inputevents_CV t INNER JOIN d_items d on c.itemid=d.itemid',
                'd.label',
                't.amount',
                't.amountuom',
                'WHERE t.amount is NOT NULL AND t.amountuom is NOT NULL'
            ],
            [
                'inputevents_MV t INNER JOIN d_items d on t.itemid=d.itemid',
                'd.label',
                't.amount',
                't.amountuom',
                'WHERE t.amount is NOT NULL AND t.amountuom is NOT NULL'
            ],
        ],
    'Presc-dose_val_rx':
        [
            'prescriptions',
            'drug',
            'dose_val_rx',
            'dose_unit_rx',
            'WHERE dose_val_rx is NOT NULL AND dose_unit_rx is NOT NULL'
        ],
    'ICU Out-los': 
        [
            'icustays',
            None,
            'los',
            None,
            'WHERE los is NOT NULL AND los !=0'
        ],
    'Transfer Out-los': 
        [
            'icustays',
            None,
            'los',
            None,
            'WHERE los is NOT NULL AND los !=0'
        ]
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
        'itemid',
        'amount',
        'amountuom',
        'rate',
        'rateuom',
        'charttime',
        'stopped'
    ],
    # stopped - D/C'd, Stopped, NotStopd, Restart, NULL
    # TODO whether to keep all
    'Input MV': [
        'icustay_id',
        'itemid',
        'amount',
        'amountuom',
        'rate',
        'rateuom',
        'starttime',
        # Metavision specific
        'statusdescription'
    ],
    # TODO whether to keep all
    # statusdescription: Changed, Paused, FinishedRunning,
    # Stopped, Rewritten, Flushed
    'Presc Start': [
        'startdate',
        #'drug_type',
        'drug'],
    'Presc End': [
        'enddate',
        #'drug_type',
        'drug']
}

TABLES = {
    # Patient tracking events
    'admissions',
    'icustays',
    'callout',
    'transfers',
    # ICU Events
    'chartevents',
    'inputevents_cv',
    'inputevents_mv',
    'datatimeevents',
    'outputevents',
    'procedureevents_mv',
    # Hospital Data
    'cptevents',
    'diagnoses_icd',  # at the end
    'drgcodes',  # at the end
    'labevents',
    'microbiologyevents',
    'notevents',
    'prescriptions',  # start and end date
    'procedures_icd',  # at the end
    'services'}

IGNORE_COLS = [
    'dbsource',
    'icustay_id',
    'pi_number',
    'pi_info',
    'pi_stage',
    # labs and inteventions
    'intervention-count'
    'vitals-mean',
    'vitals-count',
    'vitals-std',
    'icu-time',
    'count_Q',
    'std_Q',
    'cptdesc',
]

FLOAT_COLS = [  # 'chartevents-valuenum',
                # 'icustays-los', #length of stay
                'Input-amount',
                'Input-rate',
                # 'labevents-valuenum',
                'Microbiology-dilution_value',
                # 'microbiologyevents-isolate_num' #SMAILLINT,
                'Output-value',
                # 'Transfer Out-los',
]
