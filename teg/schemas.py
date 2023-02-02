# {<table_name>: [<columns>]}

# attributes of an event, not used for comparison
EVENT_IDs = ['id', 'type', 't', 'i']

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


# {<event_name>: [<event_number>, <event_table>, <time_column>]}
EVENTS = {
    # Patient tracking events
    'admissions': [1, 'admissions', 'admittime'],
    'discharges': [2, 'admissions', 'dischtime'],
    'icu_in': [3, 'icustays', 'intime'],
    'icu_out': [3, 'icustays', 'outtime'],
    'callout': [5, 'callout', 'outcometime'],
    'transfer_in': [6, 'transfers', 'intime'],
    'transfer_out': [7, 'transfers', 'outtime'],

    # ICU Events
    'chartevents': [8, 'chartevents', 'charttime'],
    # charttime preferred over storetime
    'inputevents_cv': [9, 'inputevents_cv', 'charttime'],
    # ignored endtime
    'inputevents_mv': [9, 'inputevents_mv', 'starttime'],
    # 'datatimeevents',
    # charttime preferred over storetime
    'outputevents': [10, 'outputevents', 'charttime'],
    # 'procedureevents_mv',

    # Hospital Data
    # Current Procedural Terminology
    'cptevents': [11, 'cptevents', 'chartdate'],
    # at the end after discharge, take discharge time from admissions table
    'diagnoses_icd': [12, 'diagnoses_icd', 'dischtime'],
    # at the end after discharge, take discharge time from admissions table
    'drgcodes': [13, 'drgcodes', 'dischtime'],
    'labevents': [14, 'labevents', 'charttime'],
    # charttime is NULL when unknown. Hence we use chartdate.
    # There are 41772 null charttime out of 631726 which is 6.6%.
    'microbiologyevents': [15, 'microbiologyevents', 'chartdate'],
    # TODO exclude notes with ISERROR=1
    # 886 ISERROR=1 out of 2083180, that is, around 0.04%
    # charttime preferred over storetime
    # 'notevents': [16, 'noteevents', 'charttime'],
    'prescriptions_start': [17, 'prescriptions', 'startdate'],
    'prescriptions_end': [18, 'prescriptions', 'enddate'],
    # at the end after discharge, take discharge time from admissions table
    'procedures_icd': [19, 'procedures_icd', 'dischtime'],
    'services': [20, 'services', 'transfertime']}

# {<event_name>: [<event_number>, <event_table>, <time_column>]}
LOW_FREQ_EVENTS = {
    # Patient tracking events
    'admissions': [1, 'admissions', 'admittime'],
    'discharges': [2, 'admissions', 'dischtime'],
    'icu_in': [3, 'icustays', 'intime'],
    'icu_out': [4, 'icustays', 'outtime'],
    'callout': [5, 'callout', 'outcometime'],
    'transfer_in': [6, 'transfers', 'intime'],
    'transfer_out': [7, 'transfers', 'outtime'],

    # ICU Events
    # 'chartevents': [8, 'chartevents', 'charttime'],
    # charttime preferred over storetime
    # 'inputevents_cv': [9, 'inputevents_cv', 'charttime'],
    # ignored endtime
    # 'inputevents_mv': [9, 'inputevents_mv', 'starttime'],
    # 'datatimeevents',
    # charttime preferred over storetime
    # 'outputevents': [10, 'outputevents', 'charttime'],
    # 'procedureevents_mv',

    # Hospital Data
    # Current Procedural Terminology
    'cptevents': [11, 'cptevents', 'chartdate'],
    # at the end after discharge, take discharge time from admissions table
    # 'diagnoses_icd': [12, 'diagnoses_icd', 'dischtime'],
    # at the end after discharge, take discharge time from admissions table
    # 'drgcodes': [13, 'drgcodes', 'dischtime'],
    # 'labevents': [14, 'labevents', 'charttime'],
    # charttime is NULL when unknown. Hence we use chartdate.
    # There are 41772 null charttime out of 631726 which is 6.6%.
    # 'microbiologyevents': [15, 'microbiologyevents', 'chartdate'],
    # TODO exclude notes with ISERROR=1
    # 886 ISERROR=1 out of 2083180, that is, around 0.04%
    # charttime preferred over storetime
    # 'notevents': [16, 'noteevents', 'charttime'],
    # 'prescriptions_start': [17, 'prescriptions', 'startdate'],
    # 'prescriptions_end': [18, 'prescriptions', 'enddate'],
    # at the end after discharge, take discharge time from admissions table
    # 'procedures_icd': [19, 'procedures_icd', 'dischtime'],
    'services': [20, 'services', 'transfertime']}

# {event_name : [[<include_columns>], [<exclude_columns>]]}
# ROW_ID will be excluded from all events
EVENT_COLS_EXCLUDE = {
    'admissions': [
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

    'discharges': [
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

    'icu_in': [
        'icustay_id',
        'outtime',
        'last_careunit',
        'los',
        'first_wardid',
        'last_wardid'],

    'icu_out': [
        'icustay_id',
        'intime',
        'first_careunit',
        'first_wardid',
        'last_wardid'],

    'callout': [
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
    'transfer_in': [
        'eventtype',
        'icustay_id',
        'prev_wardid',
        'curr_wardid'
        'outtime'],

    'transfer_out': [
        'icustay_id',
        'eventtype',
        'prev_wardid',
        'curr_wardid'
        'intime',
        'los'],

    'chartevents': [
        'warning',  # TODO Metavision - excluded if 1
        'error',  # TODO Metavision specific - excluded if 1
        'resultstatus',  # TODO CareVue - excluded
        'stopped'  # TODO CareVue specific - excluded if 'D/C'd'
    ],
    'cptevents': [
        'ticket_id_seq',
        'description',
        'cpt_number',
        'cpt_suffix'],

    'prescriptions_start': [
        'enddate'],

    'prescriptions_end': [
        'startdate'],

    'outputevents': [
        'cgid',
        'storetime',
    ]}

EVENT_COLS_INCLUDE = {
    'inputevents_cv': [
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
    'inputevents_mv': [
        'icustay_id',
        'itemid',
        'amount',
        'amountuom',
        'rate',
        'rateuom',
        'starttime',
        # Metavision specific
        'statusdescription'
    ]
    # TODO whether to keep all
    # statusdescription: Changed, Paused, FinishedRunning,
    # Stopped, Rewritten, Flushed
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
    'intervention-count'
    'vitals-mean',
    'vitals-count',
    'vitals-std',
    'icu-time',
    'count_Q',
    'std_Q'
]

FLOAT_COLS = [  # 'chartevents-valuenum',
                # 'icustays-los', #length of stay
                'inputevents-amount',
                'inputevents-rate',
                # 'labevents-valuenum',
                'microbiologyevents-dilution_value',
                # 'microbiologyevents-isolate_num' #SMAILLINT,
                'outputevents-value',
                # 'transfer_out-los',

]
