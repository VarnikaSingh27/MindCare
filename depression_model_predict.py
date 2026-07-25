import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

CSV_PATH = "anxiety_depression_data.csv"
SHAP_PNG = "static/shap_depression.png"
TOP_SHAP_N = 8

NUMERIC_COLS = {
    "Age": (18, 80),
    "Sleep_Hours": (2, 12),
    "Social_Support_Score": (1, 10),
    "Anxiety_Score": (0, 20),
    "Stress_Level": (0, 10),
    "Financial_Stress": (0, 10),
    "Work_Stress": (0, 10),
    "Self_Esteem_Score": (0, 10),
    "Life_Satisfaction_Score": (0, 10),
    "Loneliness_Score": (0, 10),
}

CATEGORICAL_COLS = {
    "Gender": ["Male", "Female", "Other"],
    "Family_History_Mental_Illness": ["Yes", "No"],
    "Substance_Use": ["None", "Occasional", "Frequent"],
}

def depression_class(score):
    if score <= 7: return "Low"
    if score <= 14: return "Medium"
    return "High"

def load_dataset():
    df = pd.read_csv(CSV_PATH)
    return df

def preprocess(df):
    df = df.copy()

    # numeric
    for col, (lo, hi) in NUMERIC_COLS.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median()).clip(lo, hi)

    # categorical
    for col in CATEGORICAL_COLS.keys():
        df[col] = df[col].fillna("Missing").astype(str)

    df["Class"] = df["Depression_Score"].apply(depression_class)
    label_encoder = LabelEncoder()
    df["y"] = label_encoder.fit_transform(df["Class"])

    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

    ohe_arr = ohe.fit_transform(df[list(CATEGORICAL_COLS.keys())])
    ohe_cols = ohe.get_feature_names_out(list(CATEGORICAL_COLS.keys()))
    ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols)

    X = pd.concat([df[list(NUMERIC_COLS.keys())], ohe_df], axis=1)
    y = df["y"]

    return X, y, ohe, label_encoder, list(X.columns)

def train_model():
    df = load_dataset()
    X, y, ohe, label_encoder, feature_cols = preprocess(df)

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss"
    )
    model.fit(X, y)

    return model, X, ohe, label_encoder, feature_cols

MODEL, X_TRAIN, OHE, LABEL_ENCODER, FEATURE_COLS = train_model()

# ===========================================================
# Prediction (NO summarization here)
# ===========================================================

def predict_depression(user_input, phq_answers):
    """
    user_input = dict from frontend
    phq_answers = list of 9 integers (0–3)
    """
    user_df = pd.DataFrame([user_input])

    # numeric
    for col in NUMERIC_COLS.keys():
        user_df[col] = float(user_input.get(col))

    # categorical
    for col in CATEGORICAL_COLS.keys():
        user_df[col] = str(user_input.get(col))

    # OHE
    ohe_arr = OHE.transform(user_df[list(CATEGORICAL_COLS.keys())])
    ohe_cols = OHE.get_feature_names_out(list(CATEGORICAL_COLS.keys()))
    ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols)

    X_user = pd.concat([user_df[list(NUMERIC_COLS.keys())], ohe_df], axis=1)

    for c in FEATURE_COLS:
        if c not in X_user:
            X_user[c] = 0

    X_user = X_user[FEATURE_COLS]

    # prediction
    proba = MODEL.predict_proba(X_user)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = LABEL_ENCODER.inverse_transform([pred_idx])[0]

    # PHQ-9
    total = sum(phq_answers)
    if total <= 4: phq_level = "None–Minimal"
    elif total <= 9: phq_level = "Mild"
    elif total <= 14: phq_level = "Moderate"
    elif total <= 19: phq_level = "Moderately Severe"
    else: phq_level = "Severe"

    # LONG TEXT ONLY (no summarizer here)
    long_input = (
        f"The model predicts your depression level as {pred_label}. "
        f"Your PHQ-9 score indicates {phq_level} symptoms. "
        f"Age: {user_input['Age']}, Sleep: {user_input['Sleep_Hours']} hours, "
        f"Stress level: {user_input['Stress_Level']}. "
        f"These combined insights help describe your emotional health."
    )

    return {
        "prediction": pred_label,
        "probabilities": proba.tolist(),
        "phq_level": phq_level,
        "phq_total": total,
        "long_text": long_input
    }