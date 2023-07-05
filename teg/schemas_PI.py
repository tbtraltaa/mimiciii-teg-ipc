from teg.schemas import *


PI_EVENTS = [
    'PI Stage',
    'PI Site',
    'PI Wound Base',
    'PI Drainage',
    'PI Odor',
    'PI Cleansing',
    'PI Treatment',
    'PI Pressure-Reduce',
    #'PI position',
    'PI Skin Type',
    'PI Drainage Amount',
    'PI Surrounding Tissue',
    'PI Tunneling',
    'PI Undermining',
    'PI Dressing Status',
    'PI Depth',
    'PI Width',
    'PI Length'
    ]

PI_STAGE_ITEMIDS = (
    # CareVue
    551,
    552,
    553,
    # MetaVision
    224631,
    224965,
    224966,
    224967,
    224968,
    224969,
    224970,
    224971,
    227618,
    227619
)

PI_ITEM_LABELS = [
    '%Pressure Ulcer%',
    '%Impaired Skin%',
    '%Pressure Sore%',
    '%PressSore%',
    '%PressureReduceDevice%',
    '%Position%',
    '%Surrounding Tissue%',
    '%Undermining Present%',
    ]

#{<event_name>: [<table>, <item_col>, <value_col>, <uom_col>, <cast to dtype>, <where>]
PI_EVENTS_NUMERIC = {
    'PI Depth':{
        'table': f'{schema}.chartevents c INNER JOIN {schema}.d_items d ON c.itemid=d.itemid',
        'item_col': None,
        'value_col': 'c.value',
        'uom_col': None,
        'dtype': float,
        'where': '''
            AND (d.label similar to 'Impaired Skin Depth #%'
            OR d.label similar to 'PressSore Depth #%')
            AND TRIM(c.value) not similar to '\s*Other\/Remarks\s*'
            AND TRIM(c.value) not similar to '\s*0\s*cm'
            AND TRIM(c.value) not similar to '\s*0.0\s*cm'
            '''
        },
    'PI Width':{
        'table': f'{schema}.chartevents c INNER JOIN {schema}.d_items d ON c.itemid=d.itemid',
        'item_col': None,
        'value_col': 'c.value',
        'uom_col': None,
        'dtype': float,
        'where':
            '''
            AND (d.label similar to 'Impaired Skin Width #%'
            OR d.label similar to 'Pressure Sore #\d+ \[Width\]')
            AND TRIM(c.value) not similar to '\s*Other\/Remarks\s*'
            AND TRIM(c.value) not similar to '\s*0\s*cm'
            AND TRIM(c.value) not similar to '\s*0.0\s*cm'
            '''
            },
    'PI Length':{
        'table': f'{schema}.chartevents c INNER JOIN {schema}.d_items d ON c.itemid=d.itemid',
        'item_col': None,
        'value_col': 'c.value',
        'uom_col': None,
        'dtype': float,
        'where':
            '''
            AND d.label similar to 'Impaired Skin Length #%'
            AND TRIM(c.value) not similar to '\s*0\s*cm'
            AND TRIM(c.value) not similar to '\s*0.0\s*cm'
            '''
        },
    }

UOM = 'cm'

'''
mimic=# select count(*) from mimiciii.chartevents
    where (itemid>=551 and itemid<=579) or itemid=547;
  count
---------
 1728768
(1 row)
'''
# {event_name: d_items.label}, total: 30 types of chart events
# 1728768 chart events related to PI in CareVue
# The number of PIs per patient at a time is 3 in Carevue
# ['<itemid label>', ['chartevent values to ignore']]
# \d+ - number
# \s* - zero or more white space
# % - any characters
# TODO strip and uppercase before comparing


PI_EVENTS_CV = {
    'PI Stage': ['Pressure Sore #\d+ \[Stage\]',
                 ['\s*Other\/Remarks\s*', '\s*Unable to Stage\s*']],
    'PI Site': ['Press Sore Site #%', []],
    # numeric value as string such as 5 cm
    # not 0 numeric values. Checked using '\s*0\s*cm' and '\s*0.0\s*cm'
    'PI Depth': ['PressSore Depth #%',
                 ['Other\/Remarks', '\s*0\s*cm', '\s*0.0\s*cm']],
    # numeric value as string such as 5 cm
    # not 0 numeric values. Checked using '\s*0\s*cm' and '\s*0.0\s*cm'
    'PI Width': ['Pressure Sore #\d+ \[Width\]',
                 ['Other\/Remarks', '\s*0\s*cm', '\s*0.0\s*cm']],
    'PI Wound Base': ['PressSoreWoundBase#%', ['\s*Other\/Remarks\s*']],
    'PI Drainage': ['Pressure Sore #\d+ \[Drainage\]',
                        ['\s*Other\/Remarks\s*']],
    'PI Odor': ['Pressure Sore Odor#%',
                    ['\s*Other\/Remarks\s*', '\s*Not Applicable\s*']],
    # 'Other/Remarks' included
    'PI Cleansing': ['PressSoreCleansing#%', []],
    'PI Treatment': ['PressSoreTreatment#%', []],
    # 'Other/Remarks' included
    'PI Pressure-Reduce': ['PressureReduceDevice', []],
    # TODO strip and uppercase before comparing
    'PI Position': ['Position', []],
    }

'''
select count(*) from mimiciii.chartevents c
INNER JOIN mimiciii.d_items d on c.itemid=d.itemid
where d.category='Skin - Impairment';
  count
---------
 4961844
(1 row)
'''
# {event_name: d_items.label}, total: 271 types of chart events
# 4961844 chart events related to PI in Metavision
# The number of PIs per patient at a time is up to 10 in Metavision
# ['<itemid label>', ['chartevent values to ignore']]
# \d+ - number
# \s* - zero or more white space
# % - any characters
# TODO strip and uppercase before comparing values
PI_EVENTS_MV = {
    'PI Stage': ['Pressure Ulcer Stage #\d+',
                 [
                     'Not applicable',
                     'Unable to assess; dressing not removed',
                     'Unable to stage; wound is covered with eschar'
                 ]],
    'PI Site': ['Impaired Skin Site #%',
                ['\s*Not applicable\s*',
                 '\s*Resolved\\s*']],
    # numeric value as string
    'PI Depth': ['Impaired Skin Depth #%', ['\s*0\s*', '\s*0.0\s*']],
    # numeric value as string
    'PI Width': ['Impaired Skin Width #%', ['\s*0\s*', '\s*0.0\s*']],
    'PI Wound Base': ['Impaired Skin Wound Base #%', ['\s*Not assessed\s*']],
    'PI Drainage': ['Impaired Skin Drainage #%', ['\s*Not assessed\s*']],
    'PI Odor': ['Impaired Skin Odor #%', []],
    'PI Cleansing': ['Impaired Skin Cleanse #%', ['\s*Not applicable\s*']],
    'PI Treatment': ['Impaired Skin Treatment #%', ['\s*Not applicable\s*']],
    # TODO Replace 'SubQ Emphysema' with 'Sub Q emphysema'
    'PI Skin Type': ['Impaired Skin Type #%', ['\s*Not applicable\s*']],
    # numeric value as string
    'PI Length': ['Impaired Skin Length #%', ['\s*0\s*', '\s*0.0\s*']],
    'PI Drainage Amount': ['Impaired Skin Drainage Amount #%',
                           ['\s*Not assessed\s*']],
    'PI Surrounding Tissue': ['Surrounding Tissue #%',
                              ['\s*Not assessed\s*']],
    'PI Tunneling': ['Tunneling Present #%', ['\s*Not assessed\s*']],
    'PI Undermining': ['Undermining Present #%', ['\s*Not assessed\s*']],
    'PI Dressing Status': ['Impaired Skin  - Dressing Status #%',
                           ['\s*Not assessed\s*']],
    'PI Length': ['Impaired Skin Length #%', []],
    # The following are the itemid labels not used in Metavision
    # 'type': 'Impaired Skin #N- Type',
    # 'other': ['Dressing Status'],
    # 'location': 'Impaired Skin #N- Location',
    # 'location': ''Pressure Ulcer #N- Location
    # 'dressing status 1': 'Impaired Skin #N- Dressing Status',
    # 'dressing change': 'Impaired Skin  - Dressing Change #N'
}

PI_STAGE_MAP = {
        # CV
        'Red; unbroken': 1,
        'Partial thickness skin loss through epidermis and/or dermis; '\
        'ulcer may present as an abrasion, blister, or shallow crater': 2,
        'Partial thickness skin loss through epidermis and/or dermis; '\
        'ulcer may present  as an abrasion, blister, or shallow crater': 2,
        'Full thickness skin loss that may extend down to underlying fascia; '\
        'ulcer may have tunneling or undermining': 3,
        'Full thickness skin loss with damage to muscle, bone, or supporting '\
        'structures; tunneling or undermining may be present': 4,
        'Deep tissue injury': 5,
        # MV
        'Red, Unbroken': 1,
        'Intact,Color Chg': 1,
        'Part. Thickness': 2,
        'Through Dermis': 2,
        'Full Thickness': 3,
        'Through Fascia': 4,
        'To Bone': 4,
        'Deep Tiss Injury': 5}

STAGE_PI_MAP = {
    # CV, MV
    1: ['Red; unbroken', 'Red, Unbroken', 'Intact,Color Chg'],
    2: ['Partial thickness skin loss through epidermis and/or dermis; '\
    'ulcer may present as an abrasion, blister, or shallow crater',
    'Partial thickness skin loss through epidermis and/or dermis; '\
    'ulcer may present  as an abrasion, blister, or shallow crater',
    'Part. Thickness',
    'Through Dermis'],
    3: ['Full thickness skin loss that may extend down to underlying fascia; '\
    'ulcer may have tunneling or undermining',
    'Full Thickness'],
    4: ['Full thickness skin loss with damage to muscle, bone, or supporting '\
    'structures; tunneling or undermining may be present',
    'To Bone'],
    5: ['Deep tissue injury', 'Deep Tiss Injury']}

PI_VALUE_MAP = {
    'PI Site': {
        'Abdomen': 'Abdomen',
        'Ankle, Left': 'Ankle',
        'Ankle, Right': 'Ankle',
        'Back': 'Back',
        'Back, Lower': 'Back, Lower',
        'Back, Upper': 'Back, Upper',
        'Body, Upper': 'Body, Upper',
        'Breast': 'Breast',
        'Chest': 'Chest',
        'Coccyx': 'Coccyx',
        'Ear, Left': 'Ear',
        'Ear, Right': 'Ear',
        'Elbow, Left': 'Elbow',
        'Elbow, Right': 'Elbow',
        'Extremities, Lo': 'Extremities, Lower',
        'Facial': 'Facial' ,
        'Foot, Left': 'Foot',
        'Foot, Right': 'Foot',
        'Gluteal, Left': 'Gluteal',
        'Gluteal, Right': 'Gluteal',
        'Hand, Right': 'Hand',
        'Head': 'Head',
        'Heel, Left': 'Heal',
        'Heel, Right': 'Heal',
        'Hip, Left': 'Hip',
        'Hip, Right': 'Hip',
        'Labia': 'Labia',
        'Left knee': 'Knee',
        'Right knee': 'Knee',
        'Leg, Left Lower': 'Leg, Lower',
        'Leg, Left Upper': 'Leg, Upper',
        'Leg, Right Lower': 'Leg, Lower',
        'Leg, Right Upper': 'Leg, Upper',
        'Mouth, Left': 'Mouth',
        'Mouth, Right': 'Mouth',
        'Nare, Left': 'Nare',
        'Nare, Right': 'Nare',
        'Neck': 'Neck',
        'Occiput': 'Occiput',
        'Oral': 'Oral',
        'Penis': 'Penis',
        'PeriAnal': 'Perianal',
        'Perineum': 'Perineum',
        'Scrotom': 'Scrotom',
        'Shoulder, Right': 'Shoulder',
        'Thoracotomy, L': 'Thoracotomy',
        'Toes': 'Toes',
        # MV
        'Abdominal': 'Abdomen',
        'Ankle, Lateral-L': 'Ankle',
        'Ankle, Lateral-R': 'Ankle',
        'Ankle, Medial-L': 'Ankle',
        'Ankle, Medial-R': 'Ankle',
        'Arm, Left Lower': 'Arm',
        'Arm, Left Upper': 'Arm',
        'Arm, Right Lower': 'Arm',
        'Arm, Right Upper': 'Arm',
        # 'Back': 'Back',
        # 'Back, Lower': 'Back, Lower',
        # 'Back, Upper': 'Back, Upper',
        'Body, Lower': 'Body, Lower',
        # 'Body, Upper': 'Body, Upper',
        # 'Breast': 'Breast',
        # 'Chest': 'Chest',
        'Chest Tube #1': 'Chest Tube',
        'Chest Tube #2': 'Chest Tube',
        'Chest Tube #3': 'Chest Tube',
        'Chest Tube #4': 'Chest Tube',
        # 'Coccyx': 'Coccyx',
        # 'Ear, Left': 'Ear',
        # 'Ear, Right': 'Ear',
        # 'Elbow, Left': 'Elbow',
        # 'Elbow, Right': 'Elbow',
        'Extremities, Lower': 'Extremities, Lower',
        'Extremities, Upper': 'Extremities, Upper',
        'Eye, Left': 'Eye',
        'Eye, Right': 'Eye',
        # 'Facial': 'Facial',
        'Fingers': 'Fingers',
        'Fingers, L': 'Fingers',
        'Fingers, Left': 'Fingers',
        'Fingers, R': 'Fingers',
        'Fingers, Right': 'Fingers',
        'Fingers,L': 'Fingers',
        # 'Foot, Left': 'Foot',
        # 'Foot, Right': 'Foot',
        'Front': 'Front',
        # 'Gluteal, Left': 'Foot',
        # 'Gluteal, Right': 'Foot',
        'Groin, Left': 'Groin',
        'Groin, Right': 'Groin',
        'Hand, Left': 'Hand',
        # 'Hand, Right': 'Hand',
        # 'Head': 'Head',
        # 'Heel, Left': 'Heel',
        # 'Heel, Right': 'Heel',
        'Heel,Left': 'Hand',
        # 'Hip, Left': 'Hip',
        # 'Hip, Right': 'Hip',
        'Ischial, Left': 'Ischial',
        'Ischial, Right': 'Ischial',
        'Knee, Left': 'Knee',
        'Knee, Right': 'Knee',
        # 'Labia': 'Labia',
        #'Leg, Left Lower': 'Leg, Lower',
        #'Leg, Left Upper': 'Leg, Upper',
        #'Leg, Right Lower': 'Leg, Lower',
        #'Leg, Right Upper': 'Leg, Upper',
        'Mastoid': 'Mastoid',
        # 'Mouth, Left': 'Mouth',
        # 'Mouth, Right': 'Mouth',
        # 'Nare, Left': 'Nare',
        # 'Nare, Right': 'Nare',
        # 'Neck': 'Neck',
        'Nose': 'Nose',
        # Not applicable
        # 'Occiput': 'Occiput',
        # 'Oral': 'Oral',
        # 'Penis': 'Penis',
        'Perianal': 'Perianal',
        # 'Perineum': 'Perineum',
        'Resolved': 'Resolved',
        'Sacrum': 'Sacrum',
        'Scapula, Left': 'Scapula',
        'Scapula, Right': 'Scapula',
        'Scrotum': 'Scrotum',
        'Shoulder, Left': 'Shoulder',
        # 'Shoulder, Right': 'Shoulder',
        'Sternal': 'Sternal',
        'Thigh, Left': 'Thigh',
        'Thigh, Right': 'Thigh',
        'Thorocotomy, Left': 'Thorocotomy',
        'Thorocotomy, Right': 'Thorocotomy',
        'Toes, Left': 'Toes',
        'Toes, Right': 'Toes',
        'Torso': 'Torso'},
    'PI Odor': {
        'Negative': 'Negative',
        'Positive': 'Positive',
        '0': 'Negative',
        '1': 'Positive'},
    'PI Treatment': {
        # CV
        'Accuzyme': 1,
        'Ace Wrap': 2,
        'Adaptic': 3,
        'Alginate/Kalstat': 4,
        'Allevyn': 'Allevyn Foam Dressing',
        'Allevyn Trach': 5,
        'Antifungal Oint': 'Antifungal Oint',
        'Antifungal Powde': 'Antifungal Oint',
        'Aquacel': 'Aquacel',
        'Aquacel AG': 'Aquacel',
        'Aquaphor': 'Aquaphor',
        'Collagenase': 'Collagenase',
        'Coloplast': 'Coloplast',
        'Dermagran': 'Dermagran',
        'Dry Sterile Dsg': 'Dry Sterile Dressing',
        'Duoderm': 'Duoderm',
        'Hydrogel/Vigilon': 'Hydrogel/Vigilon',
        'Open to Air': 'Open to Air',
        'Other/Remarks': 'Other/Remarks',
        'Transparent': 'Transparent',
        'Wet to Dry': 'Wet to Dry',
        'Wnd Gel/Allevyn': 'Wound Gel',
        'Wound Gel': 'Wound Gel',
        'Wound Gel/Adapti': 'Wound Gel',
        # MV
        'Accuzyme (enzymatic debrider)': 'Accuzyme',
        'Adaptic': 'Adaptic',
        'Allevyn Foam Dressing': 'Allevyn Foam Dressing',
        'Aloe Vesta Anti-Fungal': 'Antifungal Oint',
        'Aloe Vesta Anti-Fungal Ointment': 'Antifungal Oint',
        'Aquacel AG Rope': 'Aquacel',
        'Aquacel AG Sheet': 'Aquacel',
        'Aquacel Rope': 'Aquacel',
        'Aquacel Sheet 4 x 4': 'Aquacel',
        'Aquacel Sheet 6 x 6': 'Aquacel',
        'Aquaphor': 'Aquaphor',
        'Collagenese (Santyl-enzymatic)': 'Collagenase',
        'Dakins Solution': 'Dakins Solution',
        'Double Guard Ointment': 'Double Guard Ointment',
        'Drainage Bag': 'Drainage Bag',
        #'Drainage Bag',
        'Dry Sterile Dressing': 'Dry Sterile Dressing',
        'Duoderm CGF': 'Duoderm',
        'Duoderm Extra Thin': 'Duoderm',
        'Duoderm Gel': 'Duoderm',
        'Iodoform': 'Iodoform',
        'Iodoform Gauze': 'Iodoform',
        'Mepilex Foam Dressing': 'Mepilex Foam Dressing',
        'Mesait': 'Mesait',
        'NU-Gauze': 'NU-Gauze',
        'None-Open to Air': 'Open to Air',
        #'Not applicable',
        #'Not applicable ',
        'Softsorb': 'Softsorb',
        'Telfa': 'Telfa',
        'Therapeutic  Ointment': 'Therapeutic Ointment',
        'Therapeutic Ointment': 'Therapeutic Ointment',
        'Transparent': 'Transparent',
        'VAC-White Foam': 'VAC-White Foam',
        'VAC-black dsg': 'VAC-Black Dsg',
        'VAC-white foam': 'VAC-White Foam',
        'Vaseline Gauze': 'Vaseline Gauze',
        'Vigilon Sheet Gel': 'Vigilon Sheet Gel',
        'Wet to Dry': 'Wet to Dry',
        'Xeroform': 'Xeroform'},
}

PI_VALUE_MAP_OLD = {
    'PI Site': {
        # CV
        'Abdomen': 1,
        'Ankle, Left': 2,
        'Ankle, Right': 2,
        'Back': 3,
        'Back, Lower': 4,
        'Back, Upper': 5,
        'Body, Upper': 6,
        'Breast': 7,
        'Chest': 8,
        'Coccyx': 9,
        'Ear, Left': 10,
        'Ear, Right': 10,
        'Elbow, Left': 11,
        'Elbow, Right': 11,
        'Extremities, Lo': 12,
        'Facial': 13,
        'Foot, Left': 14,
        'Foot, Right': 14,
        'Gluteal, Left': 15,
        'Gluteal, Right': 15,
        'Hand, Right': 16,
        'Head': 17,
        'Heel, Left': 18,
        'Heel, Right': 18,
        'Hip, Left': 19,
        'Hip, Right': 19,
        'Labia': 20,
        'Left knee': 21,
        'Right knee': 21,
        'Leg, Left Lower': 22,
        'Leg, Left Upper': 23,
        'Leg, Right Lower': 22,
        'Leg, Right Upper': 23,
        'Mouth, Left': 24,
        'Mouth, Right': 24,
        'Nare, Left': 25,
        'Nare, Right': 25,
        'Neck': 26,
        'Occiput': 27,
        'Oral': 28,
        'Penis': 20,
        'PeriAnal': 30,
        'Perineum': 31,
        'Scrotom': 32,
        'Shoulder, Right': 33,
        'Thoracotomy, L': 34,
        'Toes': 35,
        # MV
        'Abdominal': 1,
        'Ankle, Lateral-L': 2,
        'Ankle, Lateral-R': 2,
        'Ankle, Medial-L': 2,
        'Ankle, Medial-R': 2,
        'Arm, Left Lower': 37,
        'Arm, Left Upper': 37,
        'Arm, Right Lower': 37,
        'Arm, Right Upper': 37,
        # 'Back': 3,
        # 'Back, Lower': 4,
        # 'Back, Upper': 5,
        'Body, Lower': 38,
        # 'Body, Upper': 6,
        # 'Breast': 7,
        # 'Chest': 8,
        'Chest Tube #1': 39,
        'Chest Tube #2': 39,
        'Chest Tube #3': 39,
        'Chest Tube #4': 39,
        # 'Coccyx': 9,
        # 'Ear, Left': 10,
        # 'Ear, Right': 10,
        # 'Elbow, Left': 11,
        # 'Elbow, Right': 11,
        'Extremities, Lower': 12,
        'Extremities, Upper': 40,
        'Eye, Left': 41,
        'Eye, Right': 41,
        # 'Facial': 13,
        'Fingers': 42,
        'Fingers, L': 42,
        'Fingers, Left': 42,
        'Fingers, R': 42,
        'Fingers, Right': 42,
        'Fingers,L': 42,
        # 'Foot, Left': 14,
        # 'Foot, Right': 14,
        'Front': 43,
        # 'Gluteal, Left': 15,
        # 'Gluteal, Right': 15,
        'Groin, Left': 44,
        'Groin, Right': 44,
        'Hand, Left': 16,
        # 'Hand, Right': 16,
        # 'Head': 17,
        # 'Heel, Left': 18,
        # 'Heel, Right': 18,
        'Heel,Left': 18,
        # 'Hip, Left': 19,
        # 'Hip, Right': 19,
        'Ischial, Left': 45,
        'Ischial, Right': 45,
        'Knee, Left': 21,
        'Knee, Right': 21,
        # 'Labia': 20,
        # 'Leg, Left Lower': 22,
        # 'Leg, Left Upper': 23,
        # 'Leg, Right Lower': 22,
        # 'Leg, Right Upper': 23,
        'Mastoid': 46,
        # 'Mouth, Left': 24,
        # 'Mouth, Right': 24,
        # 'Nare, Left': 25,
        # 'Nare, Right': 25,
        # 'Neck': 26,
        'Nose': 47,
        # Not applicable
        # 'Occiput': 27,
        # 'Oral': 28,
        # 'Penis': 20,
        'Perianal': 30,
        # 'Perineum': 31,
        'Resolved': 0,
        'Sacrum': 48,
        'Scapula, Left': 49,
        'Scapula, Right': 49,
        'Scrotum': 32,
        'Shoulder, Left': 33,
        # 'Shoulder, Right': 33,
        'Sternal': 50,
        'Thigh, Left': 51,
        'Thigh, Right': 51,
        'Thorocotomy, Left': 34,
        'Thorocotomy, Right': 34,
        'Toes, Left': 35,
        'Toes, Right': 36,
        'Torso': 52},
    'PI Odor': {
        'Negative': 0,
        'Positive': 1,
        '0': 0,
        '1': 1},
    'PI Treatment': {
        # CV
        'Accuzyme': 1,
        'Ace Wrap': 2,
        'Adaptic': 3,
        'Alginate/Kalstat': 4,
        'Allevyn': 5,
        'Allevyn Trach': 5,
        'Antifungal Oint': 6,
        'Antifungal Powde': 6,
        'Aquacel': 7,
        'Aquacel AG': 7,
        'Aquaphor': 8,
        'Collagenase': 9,
        'Coloplast': 10,
        'Dermagran': 11,
        'Dry Sterile Dsg': 12,
        'Duoderm': 13,
        'Hydrogel/Vigilon': 14,
        'Open to Air': 15,
        'Other/Remarks': 16,
        'Transparent': 17,
        'Wet to Dry': 18,
        'Wnd Gel/Allevyn': 20,
        'Wound Gel': 20,
        'Wound Gel/Adapti': 20,
        # MV
        'Accuzyme (enzymatic debrider)': 1,
        'Adaptic': 3,
        'Allevyn Foam Dressing': 5,
        'Aloe Vesta Anti-Fungal': 6,
        'Aloe Vesta Anti-Fungal Ointment': 6,
        'Aquacel AG Rope': 7,
        'Aquacel AG Sheet': 7,
        'Aquacel Rope': 7,
        'Aquacel Sheet 4 x 4': 7,
        'Aquacel Sheet 6 x 6': 7,
        'Aquaphor': 8,
        'Collagenese (Santyl-enzymatic)': 9,
        'Dakins Solution': 21,
        'Double Guard Ointment': 22,
        'Drainage Bag': 23,
        #'Drainage Bag',
        'Dry Sterile Dressing': 12,
        'Duoderm CGF': 13,
        'Duoderm Extra Thin': 13,
        'Duoderm Gel': 13,
        #'Duoderm Gel',
        'Iodoform': 24,
        'Iodoform Gauze': 24,
        'Mepilex Foam Dressing': 6,
        'Mesait': 25,
        'NU-Gauze': 26,
        'None-Open to Air': 15,
        #'Not applicable',
        #'Not applicable ',
        'Softsorb': 27,
        'Telfa': 28,
        'Therapeutic  Ointment': 29,
        'Therapeutic Ointment': 29,
        'Transparent': 17,
        'VAC-White Foam': 30,
        'VAC-black dsg': 31,
        'VAC-white foam': 32,
        'Vaseline Gauze': 33,
        'Vigilon Sheet Gel': 34,
        'Wet to Dry': 18,
        'Xeroform': 35},
    'Braden Activity': {
        # CV
        'Bedfast': 1,
        'Chairfast': 2,
        'Walks Frequently': 3,
        'Walks Occasional': 4,
        # MV
        #'Bedfast': 1,
        #'Chairfast': 2,
        #'Walks Frequently': 3,
        'Walks Occasionally': 4,},
    'Braden Friction/Shear': {
        # CV
        'No Apparent Prob': 1,
        'Potential Prob': 2,
        'Problem': 3,
        # MV
        'No Apparent Problem': 1,
        'Potential Problem': 2,
        #'Problem': 3
        },
    'Braden Mobility': {
        # CV
        'No Limitations': 1,
        'Sl. Limited': 2,
        'Very Limited': 3,
        'Comp. Immobile': 4,
        # MV
        'No Limitations': 1,
        'Slight Limitations': 2,
        #'Very Limited': 3,
        'Completely Immobile': 4},
    'Braden Moisture': {
        # CV
        'Consist. Moist': 1,
        'Moist': 2,
        'Occ. Moist': 3,
        'Rarely Moist': 4,
        # MV
        'Consistently Moist': 1,
        #'Moist': 2,
        'Occasionally Moist': 3,
        #'Rarely Moist': 4
        },
    'Braden Nutrition': {
        # CV
        'Adequate': 1,
        'Excellent': 2,
        'Prob. Inadequate': 3,
        'Very Poor': 4,
        # MV
        #'Adequate': 1,
        #'Excellent': 2,
        'Probably Inadequate': 3,
        #'Very Poor': 4,
        },
    'Braden Sensory Perception': {
        'Comp. Limited': 1,
        'No Impairment': 2,
        'Sl. Limited': 3,
        'Very Limited': 4,
        # MV
        'Completely Limited': 1,
        #'No Impairment': 2,
        'Slight Impairment': 3,
        #'Very Limited': 4
        }
}
