PI_VITALS = [
                'white blood cell count',
                'magnesium',
                'creatinine',
                'oxygen saturation',
                'platelets',
                'blood urea nitrogen',
                'sodium',
                'hematocrit',

                'glascow coma scale total',
                'prothrombin time pt',
                'partial pressure of carbon dioxide',
                'ph',

                'partial pressure of oxygen',
                'troponin-t',
                'lactate',
                'lactic acid',
                'albumin',
                'bilirubin',
                'ph urine',
                'systemic vascular resistance',

        ]
'''
'mean blood pressure',
'hemoglobin',
'mean corpuscular hemoglobin concentration',
'mean corpuscular hemoglobin',
'neutrophils',
'glucose',
'potassium',
'potassium serum',
'creatinine urine',
'creatinine ascites',
'creatinine pleural',
'creatinine body fluid',
'albumin urine',
'albumin ascites',
'albumin pleural',
'prothrombin time inr',
'total protein',
'total protein urine',
'weight',
'cardiac index',
'cholesterol'
'''
'''
PI_VITALS = [
                #'white blood cell count',
                'magnesium',
                #'creatinine',
                #'glascow coma scale total',
                #'prothrombin time pt'
            ]
'''
'''
PI_VITALS = [
                'glascow coma scale total',
                'mean blood pressure',
                'oxygen saturation',
                'partial pressure of oxygen',
                'partial pressure of carbon dioxide',
                'hemoglobin',
                'mean corpuscular hemoglobin concentration',
                'mean corpuscular hemoglobin',
                'hematocrit',
                'white blood cell count',
                'white blood cell count urine',
                'neutrophils',
                'platelets',
                'glucose',
                'sodium',
                'potassium',
                'potassium serum',
                'creatinine',
                'creatinine urine',
                'creatinine ascites',
                'creatinine pleural',
                'creatinine body fluid',
                'blood urea nitrogen',
                'albumin',
                'albumin urine',
                'albumin ascites',
                'albumin pleural',
                'bilirubin',
                'troponin-t',
                'prothrombin time inr',
                'prothrombin time pt',
                'total protein',
                'total protein urine',
                'weight',
                'ph',
                'ph urine',
                'magnesium',
                'lactate',
                'lactic acid',
                'cardiac index',
                'systemic vascular resistance',
                'cholesterol'
                #'calcium',
                #'phosphorous',
                #'co2',
                #'co2 (etco2, pco2, etc.)',
                #'central venous pressure',
                #'red blood cell count urine',
                #'cholesterol hdl',
                #'cholesterol ldl',
                #'chloride urine',
                #'red blood cell count csf',
                #'lymphocytes body fluid',
                #'red blood cell count ascites',
                #'red blood cell count pleural',
                #'calcium urine',
                #'systolic blood pressure',
                #'diastolic blood pressure',
                #'temperature',
                #'central venous pressure',
        ]
'''
# name : [[ICD9_codes], to count the ICD9 code for current_admission]
# all true because we use only the first admissions
CHRONIC_ILLNESS = {
    'Diabetes': [['250%'], True],
    'Neuropathy': [['356%', '357%'], True],
    'Peripheral vascular disease': [['443%'], True],
    'Amputation': [['88%'], True],
    'Spinal cord injury': [['95200', '95205', '95210', '95215', '9522', '9528', '9539', '9523'], True], 
    'Coronary artery disease': [['440%'], True],
    'Leukemia': [['204%', '205%', '206%', '207%', '208%'], True],
    'Stroke': [['43491%'], True],
    'Congestive heart failure': [['428%'], True],
    'Anemia': [['2800', '2859', '281%'], True],
    'Urinary incontinence': [['7883%'], True],
    'Incontinence of feces':[['7876%'], True]
    # additional
    #'History of TIA/stroke w/o resid': ['V1254'] ,
    #'Family history of diabetes mellitus': ['V180'],
    #'Family history of stroke': ['V171'],
    }