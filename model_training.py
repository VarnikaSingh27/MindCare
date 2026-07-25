# model_training.py
import os
import joblib
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

MODEL_DIR = os.getenv("MODEL_DIR", "./saved_models")

def preprocess_features(df):
    df_processed = df.copy()

    categorical_cols = [
        'Gender', 'Marital_Status', 'Smoking_Habit',
        'Meditation_Practice', 'Exercise_Type'
    ]

    numerical_cols = [
        'Age', 'Sleep_Duration', 'Sleep_Quality', 'Physical_Activity',
        'Screen_Time', 'Caffeine_Intake', 'Alcohol_Intake',
        'Work_Hours', 'Travel_Time', 'Social_Interactions',
        'Blood_Pressure', 'Cholesterol_Level', 'Blood_Sugar_Level'
    ]

    # Ensure numerical columns exist
    for c in numerical_cols:
        if c in df_processed.columns:
            df_processed[c] = pd.to_numeric(df_processed[c], errors='coerce').fillna(df_processed[c].median())
        else:
            # if missing, create zeros
            df_processed[c] = 0

    # label encode target
    target_encoder = LabelEncoder()
    df_processed['Stress_Encoded'] = target_encoder.fit_transform(df_processed['Stress_Detection'])
    y = df_processed['Stress_Encoded']

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_present = [c for c in categorical_cols if c in df_processed.columns]
    ohe_data = ohe.fit_transform(df_processed[cat_present])

    ohe_df = pd.DataFrame(
        ohe_data,
        columns=ohe.get_feature_names_out(cat_present),
        index=df_processed.index
    )

    X = pd.concat([df_processed[numerical_cols], ohe_df], axis=1)

    feature_columns = list(X.columns)
    encoders = {"OneHotEncoder": ohe, "ohe_columns": cat_present}

    return X, y, encoders, target_encoder, feature_columns

def train_xgb(X, y, target_encoder):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        reg_alpha=0.5,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    try:
        print("Accuracy (holdout):", acc)
        print(classification_report(y_te, y_pred, target_names=target_encoder.classes_))
    except Exception:
        pass
    return model, X_tr

def save_artifacts(model, encoders, target_encoder, feature_columns, X_train_sample):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.joblib"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.joblib"))
    joblib.dump(target_encoder, os.path.join(MODEL_DIR, "target_encoder.joblib"))
    joblib.dump(X_train_sample, os.path.join(MODEL_DIR, "X_train_sample.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f)