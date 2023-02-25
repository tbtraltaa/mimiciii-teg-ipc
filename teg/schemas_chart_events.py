from teg.schemas import *

# {event_name: d_items.label}
CHART_EVENTS = {
    #'Braden Activity': ['%Braden Activity%', []],
    #'Braden Friction/Shear': ['%Braden Frict%', []],
    #'Braden Mobility': ['%Braden Mobility%', []],
    #'Braden Moisture': ['%Braden Moisture%', []],
    #'Braden Nutrition': ['%Braden Nutrition%', []],
    #'Braden Sensory Perception': ['%Braden Sensory%', []],
    #'Braden Score': ['%Braden Score%', []],
    }

CHART_EVENTS_NUMERIC = {
    'Braden Score':{
        'table': f'{schema}.chartevents c INNER JOIN {schema}.d_items d ON c.itemid=d.itemid',
        'item_col': None,
        'value_col': 'c.value',
        'uom_col': None,
        'dtype':float,
        'where': 
            '''
            AND d.label similar to '%Braden Score%'
            '''
            },
    }

CHART_VALUE_MAP = {
    'Braden Activity': {
        # CV
        'Bedfast': 'Bedfast',
        'Chairfast': 'Chairfast',
        'Walks Frequently': 'Walks Frequently',
        'Walks Occasional': 'Walks Occasional',
        # MV
        'Bedfast': 'Bedfast',
        'Chairfast': 'Chairfast',
        'Walks Frequently': 'Walks Frequently',
        'Walks Occasionally': 'Walks Occasional',},
    'Braden Friction/Shear': {
        # CV
        'No Apparent Prob': 'No Apparent Problem',
        'Potential Prob': 'Potential Problem',
        'Problem': 'Problem',
        # MV
        'No Apparent Problem': 'No Apparent Problem',
        'Potential Problem': 'Potential Problem',
        #'Problem': 3
        },
    'Braden Mobility': {
        # CV
        'No Limitations': 'No Limitations',
        'Sl. Limited': 'Slight Limitations',
        'Very Limited': 'Very Limited',
        'Comp. Immobile': 'Completely Immobile',
        # MV
        #'No Limitations': 'No Limitations',
        'Slight Limitations': 'Slight Limitations',
        #'Very Limited': 'Very Limited',
        'Completely Immobile': 'Completely Immobile'},
    'Braden Moisture': {
        # CV
        'Consist. Moist': 'Consistently Moist',
        'Moist': 'Moist',
        'Occ. Moist': 'Occasionally Moist',
        'Rarely Moist': 'Rarely Moist',
        # MV
        'Consistently Moist': 'Consistently Moist',
        #'Moist': 'Moist',
        'Occasionally Moist': 'Occasionally Moist',
        #'Rarely Moist': 'Rarely Moist',
        },
    'Braden Nutrition': {
        # CV
        'Adequate': 'Adequate',
        'Excellent': 'Excellent',
        'Prob. Inadequate': 'Probably Inadequate',
        'Very Poor': 'Very Poor',
        # MV
        #'Adequate': 'Adequate',
        #'Excellent': 'Excellent',
        'Probably Inadequate': 'Probably Inadequate',
        #'Very Poor': 'Very Poor',
        },
    'Braden Sensory Perception': {
        'Comp. Limited': 'Completely Limited',
        'No Impairment': 'No Impairment',
        'Sl. Limited': 'Slight Impairment',
        'Very Limited': 'Very Limited',
        # MV
        'Completely Limited': 'Completely Limited',
        #'No Impairment': 'No Impairment',
        'Slight Impairment': 'Slight Impairment',
        #'Very Limited': 'Very Limited'
        }
    }
