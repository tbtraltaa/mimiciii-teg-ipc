PI_EVENTS = [
                    'PI stage',
                    'PI site',
                    'PI wound base',
                    'PI drainage',
                    'PI odor',
                    'PI cleansing',
                    'PI treatment',
                    'PI pressure-reduce',
                    #'PI position',
                    'PI skin type',
                    'PI drainage amount',
                    'PI surrounding tissue',
                    'PI tunneling',
                    'PI undermining',
                    'PI dressing status',
                    ]

PI_EVENTS_NUMERIC = [
                    'PI depth',
                    'PI width',
                    'PI length']
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
    'PI stage': ['Pressure Sore #\\d+ \\[Stage\\]',
                 ['\\s*Other\/Remarks\\s*', '\\s*Unable to Stage\\s*']],
    'PI site': ['Press Sore Site #%', []],
    # numeric value as string such as 5 cm
    # not 0 numeric values. Checked using '\s*0\s*cm' and '\s*0.0\s*cm'
    'PI depth': ['PressSore Depth #%',
                 ['Other\/Remarks', '\\s*0\\s*cm', '\\s*0.0\\s*cm']],
    # numeric value as string such as 5 cm
    # not 0 numeric values. Checked using '\s*0\s*cm' and '\s*0.0\s*cm'
    'PI width': ['Pressure Sore #\\d+ \\[Width\\]',
                 ['Other\/Remarks', '\\s*0\\s*cm', '\\s*0.0\\s*cm']],
    'PI wound base': ['PressSoreWoundBase#%', ['\\s*Other\/Remarks\\s*']],
    'PI drainage': ['Pressure Sore #\\d+ \\[Drainage\\]',
                        ['\\s*Other\/Remarks\\s*']],
    'PI odor': ['Pressure Sore Odor#%',
                    ['\\s*Other\/Remarks\\s*', '\\s*Not Applicable\\s*']],
    # 'Other/Remarks' included
    'PI cleansing': ['PressSoreCleansing#%', []],
    'PI treatment': ['PressSoreTreatment#%', []],
    # 'Other/Remarks' included
    'PI pressure-reduce': ['PressureReduceDevice', []],
    # TODO strip and uppercase before comparing
    'PI position': ['Position', []]}

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
    'PI stage': ['Pressure Ulcer Stage #%',
                 [
                     'Not applicable',
                     'Unable to assess; dressing not removed',
                     'Unable to stage; wound is covered with eschar'
                 ]],
    'PI site': ['Impaired Skin Site #%',
                ['\\s*Not applicable\\s*',
                 '\\s*Resolved\\s*']],
    # numeric value as string
    'PI depth': ['Impaired Skin Depth #%', ['\\s*0\\s*', '\\s*0.0\\s*']],
    # numeric value as string
    'PI width': ['Impaired Skin Width #%', ['\\s*0\\s*', '\\s*0.0\\s*']],
    'PI wound base': ['Impaired Skin Wound Base #%', ['\\s*Not assessed\\s*']],
    'PI drainage': ['Impaired Skin Drainage #%', ['\\s*Not assessed\\s*']],
    'PI odor': ['Impaired Skin Odor #%', []],
    'PI cleansing': ['Impaired Skin Cleanse #%', ['\\s*Not applicable\\s*']],
    'PI treatment': ['Impaired Skin Treatment #%', []],
    # TODO Replace 'SubQ Emphysema' with 'Sub Q emphysema'
    'PI skin type': ['Impaired Skin Type #%', []],
    # numeric value as string
    'PI length': ['Impaired Skin Length #%', ['\\s*0\\s*', '\\s*0.0\\s*']],
    'PI drainage amount': ['Impaired Skin Drainage Amount #%',
                           ['\\s*Not assessed\\s*']],
    'PI surrounding tissue': ['Surrounding Tissue #%',
                              ['\\s*Not assessed\\s*']],
    'PI tunneling': ['Tunneling Present #%', ['\\s*Not assessed\\s*']],
    'PI undermining': ['Undermining Present #%', ['\\s*Not assessed\\s*']],
    'PI dressing status': ['Impaired Skin  - Dressing Status #%',
                           ['\\s*Not assessed\\s*']]
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
        'ulcer may present as  an abrasion, blister, or shallow crater': 2,
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

PI_VALUE_MAP = {
    'PI site': {
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
    'PI odor': {
        'Negative': 0,
        'Positive': 1,
        '0': 0,
        '1': 1},
    'PI treatment': {
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
        #'Not applicable',
        'Softsorb': 27,
        'Telfa': 28,
        'Therapeutic  Ointment': 29,
        #'Therapeutic Ointment',
        'Transparent': 17,
        'VAC-White Foam': 30,
        'VAC-black dsg': 31,
        'VAC-white foam': 32,
        'Vaseline Gauze': 33,
        'Vigilon Sheet Gel': 34,
        'Wet to Dry': 18,
        'Xeroform': 35},
}
