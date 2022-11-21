# {<table_name>: [<columns>]}

IDENTIFIERS = ['subject_id', 'hadm_id']

PATIENTS = {
            'patients': [
                            'gender',
                            'dob'],
            'admissions': [
                            'insurance',
                            'language',
                            'religion',
                            'marital_status',
                            'ethnicity',
                            'diagnosis']}


#{<event_name>: [<event_number>, <event_table>, <time_column>]}
EVENTS = {
            #Patient tracking events
            'admissions': [1, 'admissions', 'admittime'],
            'discharges': [2, 'admissions', 'dischtime'],
            'icustays': [3, 'icustays', 'intime'],
            'callout': [5, 'callout', 'outcometime'],
            'transfer': [6, 'transfers', 'intime'],
            #'transfer_out': [7, 'transfers', 'outtime'],

            #ICU Events 
            'chartevents': [8, 'chartevents', 'charttime'],
            #charttime preferred over storetime
            'inputevents_cv': [9, 'inputevents_cv', 'charttime'],
            #ignored endtime
            'inputevents_mv': [9, 'inputevents_mv', 'starttime'],
            #'datatimeevents',
            #charttime preferred over storetime
            'outputevents': [10, 'outputevents', 'charttime'],
            #'procedureevents_mv',

            #Hospital Data
            #Current Procedural Terminology
            'cptevents': [11, 'cptevents', 'chartdate'],
            #at the end after discharge, take discharge time from admissions table
            'diagnoses_icd': [12, 'diagnoses_icd', 'dischtime'],
            #at the end after discharge, take discharge time from admissions table
            'drgcodes': [13, 'drgcodes', 'dischtime'], 
            'labevents': [14, 'labevents', 'charttime'],
            #charttime is NULL when unknown. Hence we use chartdate.
            #There are 41772 null charttime out of 631726 which is 6.6%.
            'microbiologyevents': [15, 'microbiologyevents', 'chartdate'],
            #TODO exclude notes with ISERROR=1
            # 886 ISERROR=1 out of 2083180, that is, around 0.04%
            #charttime preferred over storetime
            #'notevents': [16, 'noteevents', 'charttime'],
            'prescriptions_start': [17, 'prescriptions', 'startdate'],
            'prescriptions_end': [18, 'prescriptions', 'enddate'],
            #at the end after discharge, take discharge time from admissions table
            'procedures_icd': [19, 'procedures_icd', 'dischtime'],
            'services': [20, 'services', 'transfertime']}

#{<event_name>: [<event_number>, <event_table>, <time_column>]}
LOW_FREQ_EVENTS = {
            #Patient tracking events
            'admissions': [1, 'admissions', 'admittime'],
            'discharges': [2, 'admissions', 'dischtime'],
            'icustays': [3, 'icustays', 'intime'],
            'callout': [5, 'callout', 'outcometime'],
            'transfer': [6, 'transfers', 'intime'],
            #'transfer_out': [7, 'transfers', 'outtime'],

            #ICU Events 
            #'chartevents': [8, 'chartevents', 'charttime'],
            #charttime preferred over storetime
            #'inputevents_cv': [9, 'inputevents_cv', 'charttime'],
            #ignored endtime
            #'inputevents_mv': [9, 'inputevents_mv', 'starttime'],
            #'datatimeevents',
            #charttime preferred over storetime
            #'outputevents': [10, 'outputevents', 'charttime'],
            #'procedureevents_mv',

            #Hospital Data
            #Current Procedural Terminology
            'cptevents': [11, 'cptevents', 'chartdate'],
            #at the end after discharge, take discharge time from admissions table
            #'diagnoses_icd': [12, 'diagnoses_icd', 'dischtime'],
            #at the end after discharge, take discharge time from admissions table
            #'drgcodes': [13, 'drgcodes', 'dischtime'], 
            #'labevents': [14, 'labevents', 'charttime'],
            #charttime is NULL when unknown. Hence we use chartdate.
            #There are 41772 null charttime out of 631726 which is 6.6%.
            'microbiologyevents': [15, 'microbiologyevents', 'chartdate'],
            #TODO exclude notes with ISERROR=1
            # 886 ISERROR=1 out of 2083180, that is, around 0.04%
            #charttime preferred over storetime
            #'notevents': [16, 'noteevents', 'charttime'],
            #'prescriptions_start': [17, 'prescriptions', 'startdate'],
            #'prescriptions_end': [18, 'prescriptions', 'enddate'],
            #at the end after discharge, take discharge time from admissions table
            #'procedures_icd': [19, 'procedures_icd', 'dischtime'],
            'services': [20, 'services', 'transfertime']}

#{<event_name>: [<event_number>, <event_table>, <time_column>]}
HIGH_FREQ_EVENTS = {
            #Patient tracking events
            'admissions': [1, 'admissions', 'admittime'],
            'discharges': [2, 'admissions', 'dischtime'],
            'icustays': [3, 'icustays', 'intime'],
            'callout': [5, 'callout', 'outcometime'],
            'transfer': [6, 'transfers', 'intime'],

            #ICU Events 
            #'chartevents': [8, 'chartevents', 'charttime'],
            #charttime preferred over storetime
            #'inputevents_cv': [9, 'inputevents_cv', 'charttime'],
            #ignored endtime
            #'inputevents_mv': [9, 'inputevents_mv', 'starttime'],
            #'datatimeevents',
            #charttime preferred over storetime
            #'outputevents': [10, 'outputevents', 'charttime'],
            #'procedureevents_mv',

            #Hospital Data
            #Current Procedural Terminology
            'cptevents': [11, 'cptevents', 'chartdate'],
            #at the end after discharge, take discharge time from admissions table
            'diagnoses_icd': [12, 'diagnoses_icd', 'dischtime'],
            #at the end after discharge, take discharge time from admissions table
            'drgcodes': [13, 'drgcodes', 'dischtime'], 
            #'labevents': [14, 'labevents', 'charttime'],
            #charttime is NULL when unknown. Hence we use chartdate.
            #There are 41772 null charttime out of 631726 which is 6.6%.
            #'microbiologyevents': [15, 'microbiologyevents', 'chartdate'],
            #TODO exclude notes with ISERROR=1
            # 886 ISERROR=1 out of 2083180, that is, around 0.04%
            #charttime preferred over storetime
            #'notevents': [16, 'noteevents', 'charttime'],
            #'prescriptions_start': [17, 'prescriptions', 'startdate'],
            #'prescriptions_end': [18, 'prescriptions', 'enddate'],
            #at the end after discharge, take discharge time from admissions table
            'procedures_icd': [19, 'procedures_icd', 'dischtime'],
            'services': [20, 'services', 'transfertime']}

# {event_name : [[<include_columns>], [<exclude_columns>]]}
#ROW_ID will be excluded from all events
EVENT_COLS_EXCLUDE = {
            'admissions': [
                            'dischtime',
                            'discharge_location',
                            'edregtime',
                            'edouttime',
                            'deathtime', 
                            'hospical_expire_flag',
                            #exluding columns for demography
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
                            #exluding columns for demography
                            'insurance',
                            'language',
                            'religion',
                            'marital_status',
                            'ethnicity'],
            'icustays': [
                            'icustay_id',
                            'outtime',
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
                        'discharge_wardid'],
            #kept DBSOURCE which indicates CareVue or Metavision as info.
            #It is ignored when comparing events
            'transfer': [   'icustay_id',
                            'prev_wardid',
                            'curr_wardid'
                            'outtime'],
            '''
            'transfer_out': [
                            'first_careunit',
                            'first_wardid',
                            'intime'],
            '''
            'chartevents': [
                            'warning', #TODO Metavision - excluded if 1
                            'error', #TODO Metavision specific - excluded if 1
                            'resultstatus', #TODO CareVue - excluded
                            'stopped' #TODO CareVue specific - excluded if 'D/C'd',
                            ],
            'cptevents': [
                            'ticket_id_seq',
                            'description'],
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
                            #stopped - D/C'd, Stopped, NotStopd, Restart, NULL
                            #TODO whether to keep all
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
                            #TODO whether to keep all
                            #statusdescription: Changed, Paused, FinishedRunning,
                                                #Stopped, Rewritten, Flushed
                            }

TABLES = {
            #Patient tracking events
            'admissions',
            'icustays',
            'callout',
            'transfers',
            #ICU Events 
            'chartevents',
            'inputevents_cv',
            'inputevents_mv',
            'datatimeevents',
            'outputevents',
            'procedureevents_mv',
            #Hospital Data
            'cptevents',
            'diagnoses_icd', #at the end
            'drgcodes', # at the end
            'labevents',
            'microbiologyevents',
            'notevents',
            'prescriptions', #start and end date
            'procedures_icd', #at the end
            'services'}

IGNORE_COLS = [
                'dbsource',
              ]

FLOAT_COLS = [  #'chartevents-valuenum',
                #'icustays-los', #length of stay
                'inputevents-amount',
                'inputevents-rate',
                #'labevents-valuenum',
                'microbiologyevents-dilution_value',
                #'microbiologyevents-isolate_num' #SMAILLINT,
                'outputevents-value',
                #'transfer_out-los',
                'icu-mean',
                'icu-count',
                'icu-std',
                'icu-time',
                'count_Q']

EVENTS_EXCLUDE = {
        'chartevents': {'itemid': [ 211, 
                                    742,
                                    646,
                                    618,
                                    212,
                                    161,
                                    128,
                                    550,
                                    1125,
                                    220045,
                                    220210,
                                    220277,
                                    159,
                                    1484,
                                    51,
                                    8368,
                                    52,
                                    220048,
                                    227969,
                                    224650,
                                    5815,
                                    8549,
                                    5820,
                                    8554,
                                    5819,
                                    8553,
                                    834,
                                    3450,
                                    8518,
                                    3603,
                                    581,
                                    3609,
                                    8532,
                                    455,
                                    8441,
                                    456,
                                    31,
                                    5817,
                                    8551,
                                    220181,
                                    220179,
                                    220180,
                                    113,
                                    1703,
                                    220052,
                                    467,
                                    220050,
                                    220051,
                                    80,
                                    1337,
                                    674,
                                    432]

            }
        }
