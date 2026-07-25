# preprocess.py
import pandas as pd
import numpy as np
import re

# map DB column names (snake_case) -> model column names (PascalCase)
COLUMN_MAP = {
    'age': 'Age',
    'gender': 'Gender',
    'marital_status': 'Marital_Status',
    'sleep_duration': 'Sleep_Duration',
    'sleep_quality': 'Sleep_Quality',
    'physical_activity': 'Physical_Activity',
    'screen_time': 'Screen_Time',
    'caffeine_intake': 'Caffeine_Intake',
    'alcohol_intake': 'Alcohol_Intake',
    'smoking_habit': 'Smoking_Habit',
    'work_hours': 'Work_Hours',
    'travel_time': 'Travel_Time',
    'social_interactions': 'Social_Interactions',
    'meditation_practice': 'Meditation_Practice',
    'exercise_type': 'Exercise_Type',
    'blood_pressure': 'Blood_Pressure',
    'cholesterol_level': 'Cholesterol_Level',
    'blood_sugar_level': 'Blood_Sugar_Level',
    'stress_detection': 'Stress_Detection'
}

def parse_first_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    m = re.search(r'(\d+(\.\d+)?)', s)
    if m:
        try:
            return float(m.group(1))
        except:
            return np.nan
    return np.nan

def clean_yesno(x):
    if pd.isna(x):
        return 'No'
    s = str(x).strip().lower()
    if s in ('yes', 'y', 'true', '1'):
        return 'Yes'
    return 'No'

def df_db_to_model(df):
    df = df.copy()
    # only keep columns we care about + occupation (we ignore occupation for training)
    # take whichever of the keys exist in df
    present_keys = [k for k in COLUMN_MAP.keys() if k in df.columns]
    df = df[present_keys + ([c for c in ['occupation'] if c in df.columns])]
    df = df.rename(columns=COLUMN_MAP)
    # numeric coercion
    numerics = [
        'Age', 'Sleep_Duration', 'Sleep_Quality', 'Physical_Activity',
        'Screen_Time', 'Work_Hours', 'Travel_Time'
    ]
    for c in numerics:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # caffeine/alcohol/bp/chol/sugar/social interactions: parse first number
    for c in ['Caffeine_Intake','Alcohol_Intake','Blood_Pressure','Cholesterol_Level','Blood_Sugar_Level','Social_Interactions']:
        if c in df.columns:
            df[c] = df[c].apply(parse_first_number)
    # boolean-like fields
    for c in ['Smoking_Habit','Meditation_Practice']:
        if c in df.columns:
            df[c] = df[c].apply(clean_yesno)
    # categorical capitalizations
    for c in ['Gender','Marital_Status','Exercise_Type','Stress_Detection']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.capitalize()
    # drop rows with missing target
    if 'Stress_Detection' in df.columns:
        df = df[~df['Stress_Detection'].isna()]
    df = df.reset_index(drop=True)
    return df