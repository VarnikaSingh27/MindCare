# ==========================================================
# Anxiety Detection — XGBoost pipeline + GAD-7 + SHAP waterfall
# Structure mirrors the stress/XGBoost code; adapted for anxiety dataset
# File: anxiety_xgb_gad.py
# ==========================================================

import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import shap
import base64
import matplotlib.pyplot as plt

# -----------------------------
# Recommendation dictionary (reused / adapted)
# -----------------------------
recommendations_dict = {
    "Sleep_Duration_low": [
        "Try maintaining a fixed sleep schedule — go to bed and wake up at the same times daily.",
        "Avoid screens 60 minutes before bed; try reading or calm stretching instead.",
        "Reduce heavy meals and caffeine close to bedtime to improve sleep onset."
    ],
    "Sleep_Quality_low": [
        "Practice breathing or progressive muscle relaxation before bed (5–10 minutes).",
        "Make the bedroom dark, cool and quiet; use blackout curtains or white noise if needed.",
        "Limit naps and keep consistent sleep-wake windows."
    ],
    "Screen_Time_high": [
        "Take a short break every 45–60 minutes — walk or look at distant objects.",
        "Reduce late-night screen exposure and enable night mode on devices.",
        "Replace one hour of evening screen time with a relaxing activity."
    ],
    "Physical_Activity_low": [
        "Start with short 20–30 minute walks most days to reduce anxiety.",
        "Add light stretching or gentle yoga sessions to relax body and mind.",
        "Aim for consistency over intensity; small daily movement helps."
    ],
    "Caffeine_Intake_high": [
        "Reduce late afternoon caffeine and replace some cups with herbal tea.",
        "Track and limit total daily caffeine — try removing one cup per week.",
        "Avoid caffeine close to bedtime to improve sleep and reduce anxiety."
    ],
    "Alcohol_Intake_high": [
        "Limit alcohol use and avoid using it as a stress reliever.",
        "Replace some drinks with non-alcoholic calming beverages.",
        "Monitor triggers for drinking and try alternate coping strategies."
    ],
    "Cardio_high": [
        "Practice slow diaphragmatic breathing for 5 minutes daily to lower sympathetic tone.",
        "Take frequent short walks to reduce cardiovascular stress.",
        "Consult a clinician for persistent elevated heart rate or blood pressure."
    ]
}

# -----------------------------
# GAD-7 prompts (replacement for PSS)
# -----------------------------
GAD7_PROMPTS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it's hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen"
]

def get_gad7_score():
    """
    Ask GAD-7 questions (0-3). Return total, level, answers.
    """
    print("\n🧠 GAD-7 (0=Not at all, 3=Nearly every day)")
    answers = []
    for q in GAD7_PROMPTS:
        while True:
            try:
                v = int(input(f"{q} (0-3): "))
                if 0 <= v <= 3:
                    answers.append(v)
                    break
                else:
                    print("Enter 0-3 only.")
            except Exception:
                print("Invalid input, enter 0-3.")
    total = sum(answers)
    if total <= 4:
        level = "Minimal"
    elif total <= 9:
        level = "Mild"
    elif total <= 14:
        level = "Moderate"
    else:
        level = "Severe"
    print(f"GAD-7 Total: {total} → {level}")
    return total, level, answers

# -----------------------------
# Auto recommendations for anxiety dataset
# -----------------------------
def auto_recommend(user):
    recs = []

    if 'Sleep Hours' in user and float(user.get('Sleep Hours', 8)) < 6:
        recs.append(random.choice(recommendations_dict["Sleep_Duration_low"]))

    if 'Diet Quality (1-10)' in user and float(user.get('Diet Quality (1-10)', 6)) < 4:
        recs.append(random.choice(recommendations_dict["Sleep_Quality_low"]))

    if 'Screen_Time' in user and float(user.get('Screen_Time', 0)) > 4:
        recs.append(random.choice(recommendations_dict["Screen_Time_high"]))

    if 'Physical Activity (hrs/week)' in user and float(user.get('Physical Activity (hrs/week)', 0)) < 2:
        recs.append(random.choice(recommendations_dict["Physical_Activity_low"]))

    if 'Caffeine Intake (mg/day)' in user and float(user.get('Caffeine Intake (mg/day)', 0)) > 300:
        recs.append(random.choice(recommendations_dict["Caffeine_Intake_high"]))

    if 'Alcohol Consumption (drinks/week)' in user and float(user.get('Alcohol Consumption (drinks/week)', 0)) > 7:
        recs.append(random.choice(recommendations_dict["Alcohol_Intake_high"]))

    if 'Heart Rate (bpm)' in user and float(user.get('Heart Rate (bpm)', 0)) > 100:
        recs.append(random.choice(recommendations_dict["Cardio_high"]))

    if len(recs) == 0:
        recs.append("You're doing reasonably well on measurable lifestyle metrics. Continue good habits and consider small, sustainable improvements.")

    return recs

# -----------------------------
# Load dataset (anxiety_data.csv)
# -----------------------------
def load_and_preprocess_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    print("Dataset Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df

# -----------------------------
# Preprocess features (OneHot encode categorical columns except Occupation)
# Detect present columns and adapt
# -----------------------------
def preprocess_features(df):
    df_processed = df.copy()

    # Target: convert numeric Anxiety Level (1-10) to categorical labels (Low/Moderate/High)
    if "Anxiety Level (1-10)" in df_processed.columns:
        def anxiety_label(x):
            try:
                x = float(x)
            except:
                x = 0.0
            if x <= 3:
                return "Low"
            elif x <= 6:
                return "Moderate"
            else:
                return "High"
        df_processed["Anxiety_Category"] = df_processed["Anxiety Level (1-10)"].apply(anxiety_label)
        target_col = "Anxiety_Category"
    else:
        # fallback - try a few common names
        if "Anxiety_Label" in df_processed.columns:
            target_col = "Anxiety_Label"
        else:
            raise ValueError("No anxiety target column found. Expected 'Anxiety Level (1-10)' or 'Anxiety_Label'.")

    # Categorical columns to consider (omit 'Occupation' one-hot by default)
    candidate_categorical = [
        "Gender", "Smoking", "Family History of Anxiety", "Dizziness", "Medication",
        "Recent Major Life Event", "Marital_Status", "Exercise_Type", "Meditation_Practice"
    ]
    categorical_cols = [c for c in candidate_categorical if c in df_processed.columns]

    # Numerical columns (pick from likely names; only keep present ones)
    candidate_numerical = [
        "Age", "Sleep Hours", "Physical Activity (hrs/week)", "Caffeine Intake (mg/day)",
        "Alcohol Consumption (drinks/week)", "Stress Level (1-10)", "Heart Rate (bpm)",
        "Breathing Rate (breaths/min)", "Sweating Level (1-5)", "Therapy Sessions (per month)",
        "Diet Quality (1-10)", "Sleep Duration", "Sleep_Quality", "Screen_Time",
        "Work_Hours", "Travel_Time", "Social_Interactions", "Blood_Pressure",
        "Cholesterol_Level", "Blood_Sugar_Level"
    ]
    numerical_cols = [c for c in candidate_numerical if c in df_processed.columns]

    # Build target encoder
    target_encoder = LabelEncoder()
    df_processed["Target_Encoded"] = target_encoder.fit_transform(df_processed[target_col].astype(str))

    # OneHotEncode categorical columns (if any)
    encoders = {}
    if len(categorical_cols) > 0:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe_data = ohe.fit_transform(df_processed[categorical_cols])
        ohe_df = pd.DataFrame(
            ohe_data,
            columns=ohe.get_feature_names_out(categorical_cols),
            index=df_processed.index
        )
        X = pd.concat([df_processed[numerical_cols], ohe_df], axis=1)
        encoders["OneHotEncoder"] = ohe
    else:
        X = df_processed[numerical_cols].copy()

    feature_columns = list(X.columns)
    y = df_processed["Target_Encoded"].copy()

    return X, y, encoders, target_encoder, feature_columns

# -----------------------------
# Train XGBoost (multi-class)
# -----------------------------
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
        use_label_encoder=False,
        n_jobs=-1
    )

    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    print("\n✅ XGBoost trained")
    print(f"Accuracy (holdout): {acc:.3f}")
    print(classification_report(y_te, y_pred, target_names=target_encoder.classes_))

    return model, X_tr

# -----------------------------
# Predict for a user (build user_features for model)
# -----------------------------
def predict_anxiety_level_xgb(model, user_input, encoders, target_encoder, feature_columns):
    user_df = pd.DataFrame([user_input])

    # if OneHotEncoder was used
    if encoders and "OneHotEncoder" in encoders:
        categorical_cols = encoders["OneHotEncoder"].feature_names_in_.tolist() if hasattr(encoders["OneHotEncoder"], "feature_names_in_") else []
        # NOTE: Many sklearn versions store feature_names_in_ attribute; else we need to deduce input order
        # We'll just get categorical names from the encoder by reflecting transformation columns
        # But here we assume the original categorical column names used during fit are passed to encoders.
        # To be robust: if categorical cols not present in user_df, create them with default values.
        # Let's attempt to infer the categorical names used during fit:
        try:
            cat_names = encoders["OneHotEncoder"].categories_
            # categories_ is a list of arrays for each categorical feature; cannot directly extract names,
            # But OneHotEncoder exposes get_feature_names_out which we used earlier - we can reconstruct by transforming.
            ohe = encoders["OneHotEncoder"]
            categorical_input_columns = ohe.feature_names_in_.tolist() if hasattr(ohe, "feature_names_in_") else []
        except Exception:
            categorical_input_columns = []

        # Ensure categorical_input_columns exist in user_df
        for col in categorical_input_columns:
            if col not in user_df.columns:
                user_df[col] = "missing"

        # transform
        try:
            ohe = encoders["OneHotEncoder"]
            ohe_data = ohe.transform(user_df[categorical_input_columns])
            ohe_df = pd.DataFrame(
                ohe_data,
                columns=ohe.get_feature_names_out(categorical_input_columns),
                index=user_df.index
            )
        except Exception:
            # fallback: create zero columns for all feature_columns that start with categorical prefixes
            ohe_df = pd.DataFrame(0, index=user_df.index, columns=[c for c in feature_columns if "_" in c and c not in user_df.columns])

        # Numerical part: keep all feature_columns that are not in ohe_df
        numericals = [c for c in feature_columns if c not in ohe_df.columns]
        numeric_df = pd.DataFrame(index=user_df.index)
        for n in numericals:
            if n in user_df.columns:
                numeric_df[n] = user_df[n]
            else:
                # default 0 if missing
                numeric_df[n] = 0

        user_features = pd.concat([numeric_df, ohe_df], axis=1)
    else:
        # no encoder used; build numeric-only frame
        user_features = pd.DataFrame(index=user_df.index)
        for col in feature_columns:
            user_features[col] = user_df[col] if col in user_df.columns else 0

    # Ensure all feature_columns exist in user_features (order)
    for col in feature_columns:
        if col not in user_features.columns:
            user_features[col] = 0

    user_features = user_features[feature_columns]

    pred_encoded = model.predict(user_features)[0]
    proba = model.predict_proba(user_features)[0]
    pred_label = target_encoder.inverse_transform([pred_encoded])[0]

    return pred_label, proba, user_features

# -----------------------------
# SHAP: explain single user (waterfall) — using model.get_booster()
# -----------------------------
def explain_prediction_single(model, user_features, target_encoder, X_train, max_display=8):
    """
    SHAP waterfall for XGBoost multi-class model
    """
    explainer = shap.TreeExplainer(
        model.get_booster(),
        data=X_train,
        feature_perturbation="interventional",
        model_output="raw"
    )

    shap_values = explainer.shap_values(user_features)

    proba = model.predict_proba(user_features)[0]
    pred_class = int(np.argmax(proba))
    pred_label = target_encoder.classes_[pred_class]

    print(f"\n✅ SHAP waterfall for predicted class: {pred_label}")

    if isinstance(shap_values, list):
        sv_vec = np.array(shap_values[pred_class][0])
        base_raw = explainer.expected_value[pred_class]
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            sv_vec = shap_values[0, :, pred_class]
            base_raw = explainer.expected_value[pred_class]
        else:
            sv_vec = shap_values[0, :]
            base_raw = explainer.expected_value[0]
    else:
        raise TypeError("Unsupported SHAP type")

    sv_vec = np.array(sv_vec, dtype=float).flatten()
    base_scalar = float(np.array(base_raw).reshape(-1)[0])

    fv = user_features.values[0]
    fnames = np.array(user_features.columns)
    m = min(len(sv_vec), len(fv), len(fnames))
    sv_vec = sv_vec[:m]
    fv = fv[:m]
    fnames = fnames[:m]

    exp = shap.Explanation(values=sv_vec, base_values=base_scalar, data=fv, feature_names=fnames)
    shap.plots.waterfall(exp, max_display=max_display)

# -----------------------------
# Extract SHAP top features (list)
# -----------------------------
def get_shap_top_features(model, user_features, X_train, top_k=10):
    explainer = shap.TreeExplainer(
        model.get_booster(),
        data=X_train,
        feature_perturbation="interventional",
        model_output="raw"
    )
    shap_values = explainer.shap_values(user_features)

    proba = model.predict_proba(user_features)[0]
    pred_class = int(np.argmax(proba))

    if isinstance(shap_values, list):
        sv = np.array(shap_values[pred_class][0])
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            sv = shap_values[0, :, pred_class]
        else:
            sv = shap_values[0]
    else:
        raise TypeError("Unsupported SHAP type")

    sv = np.array(sv, dtype=float).reshape(-1)
    feature_list = list(zip(user_features.columns, sv))
    feature_list_sorted = sorted(feature_list, key=lambda x: abs(float(x[1])), reverse=True)

    return feature_list_sorted[:top_k]

# -----------------------------
# Subjective report generator (GAD-7 + shap top + recommendations)
# -----------------------------
def generate_subjective_report(pred_label, gad_level, user_input, shap_top_features, recommendations):
    shap_sentences = []
    for feat, val in shap_top_features[:6]:
        direction = "increases" if val > 0 else "reduces"
        shap_sentences.append(f"Feature '{feat}' (your value: {user_input.get(feat, 'N/A')}) {direction} predicted anxiety (impact={abs(val):.3f}).")
    shap_text = " ".join(shap_sentences)

    lifestyle = []
    if 'Sleep Hours' in user_input:
        lifestyle.append(f"you sleep {user_input.get('Sleep Hours')} hours/night")
    if 'Physical Activity (hrs/week)' in user_input:
        lifestyle.append(f"{user_input.get('Physical Activity (hrs/week)')} hrs/week physical activity")
    lifestyle_text = ". ".join(lifestyle)

    rec_text = " ".join([f"- {r}" for r in recommendations[:4]])

    long_text = (
        f"The XGBoost model predicted your anxiety label as {pred_label}. "
        f"The GAD-7 assessment returned {gad_level} severity. "
        f"{lifestyle_text}. "
        f"{shap_text} "
        f"Suggested actions: {rec_text}."
    )
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        out = summarizer(long_text, max_length=180, min_length=120, do_sample=False)[0]
        summary = out.get("summary_text", long_text)
    except Exception:
        summary = long_text  # fallback

    return summary

def shap_waterfall_png_base64(model, user_features, X_train):
    explainer = shap.TreeExplainer(
        model.get_booster(),
        data=X_train,
        feature_perturbation="interventional",
        model_output="raw"
    )

    shap_values = explainer.shap_values(user_features)
    proba = model.predict_proba(user_features)[0]
    pred_class = int(np.argmax(proba))

    if isinstance(shap_values, list):
        sv_vec = shap_values[pred_class][0]
        base_val = explainer.expected_value[pred_class]
    else:
        sv_vec = shap_values[0]
        base_val = explainer.expected_value

    exp = shap.Explanation(
        values=sv_vec,
        base_values=base_val,
        data=user_features.values[0],
        feature_names=user_features.columns
    )

    fig = plt.figure(figsize=(8,6))
    shap.plots.waterfall(exp, max_display=10, show=False)

    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)

    return base64.b64encode(buf.getvalue()).decode()

# ============================
# DB → ML dataframe mapping
# (adjust column names if your table uses different snake_case)
# ============================
def df_db_to_anxiety_df(df_db: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DB columns (snake_case) to ML column names (with spaces)
    that preprocess_features() expects.
    Adjust mapping keys to match your actual anxiety_data table columns.
    """
    rename_map = {
        "age": "Age",
        "gender": "Gender",
        "occupation": "Occupation",
        "sleep_hours": "Sleep Hours",
        "physical_activity_hrs": "Physical Activity (hrs/week)",
        "caffeine_intake_mg": "Caffeine Intake (mg/day)",
        "alcohol_consumption_drinks": "Alcohol Consumption (drinks/week)",
        "smoking": "Smoking",
        "family_history_anxiety": "Family History of Anxiety",
        "stress_level": "Stress Level (1-10)",
        "heart_rate_bpm": "Heart Rate (bpm)",
        "breathing_rate": "Breathing Rate (breaths/min)",
        "sweating_level": "Sweating Level (1-5)",
        "dizziness": "Dizziness",
        "medication": "Medication",
        "therapy_sessions_per_month": "Therapy Sessions (per month)",
        "recent_major_life_event": "Recent Major Life Event",
        "diet_quality": "Diet Quality (1-10)",
        "anxiety_level": "Anxiety Level (1-10)",
    }
    df = df_db.rename(columns=rename_map)
    return df

# ============================
# Train anxiety model from DB DataFrame
# ============================
def train_anxiety_from_df(df_db: pd.DataFrame):
    """
    Used by app.py startup: takes anxiety_data table as DataFrame,
    maps to ML column names, preprocesses, trains XGB and returns
    model + encoders + feature columns + a sample of X_train for SHAP.
    """
    df = df_db_to_anxiety_df(df_db)
    X, y, encoders, target_encoder, feature_columns = preprocess_features(df)
    model, X_train = train_xgb(X, y, target_encoder)
    sample = X_train.sample(min(200, len(X_train)), random_state=42)
    return model, encoders, target_encoder, feature_columns, sample

def train_and_export(csv_file_path="anxiety_data.csv"):
    df = load_and_preprocess_data(csv_file_path)
    X, y, encoders, target_encoder, feature_columns = preprocess_features(df)
    model, X_train = train_xgb(X, y, target_encoder)
    return model, encoders, target_encoder, feature_columns, X_train