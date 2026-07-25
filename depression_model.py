import os
import numpy as np
import pandas as pd
from collections import defaultdict
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

import joblib

DEP_MODEL_DIR = os.getenv("DEP_MODEL_DIR", "./saved_models_depression")

os.makedirs(DEP_MODEL_DIR, exist_ok=True)

def save_dep_artifacts(model, ohe, label_encoder, feature_columns, X_train_sample):
    joblib.dump(model, os.path.join(DEP_MODEL_DIR, "dep_xgb_model.joblib"))
    joblib.dump(ohe, os.path.join(DEP_MODEL_DIR, "dep_ohe.joblib"))
    joblib.dump(label_encoder, os.path.join(DEP_MODEL_DIR, "dep_label_encoder.joblib"))
    joblib.dump(feature_columns, os.path.join(DEP_MODEL_DIR, "dep_feature_columns.joblib"))
    joblib.dump(X_train_sample, os.path.join(DEP_MODEL_DIR, "dep_X_train_sample.joblib"))

def load_dep_artifacts():
    model = joblib.load(os.path.join(DEP_MODEL_DIR, "dep_xgb_model.joblib"))
    ohe = joblib.load(os.path.join(DEP_MODEL_DIR, "dep_ohe.joblib"))
    label_encoder = joblib.load(os.path.join(DEP_MODEL_DIR, "dep_label_encoder.joblib"))
    feature_columns = joblib.load(os.path.join(DEP_MODEL_DIR, "dep_feature_columns.joblib"))
    X_train_sample = joblib.load(os.path.join(DEP_MODEL_DIR, "dep_X_train_sample.joblib"))
    return model, ohe, label_encoder, feature_columns, X_train_sample

def train_depression_from_df(df_db: pd.DataFrame):
    df = df_db_to_depression_df(df_db)
    X, y, ohe, label_encoder, feature_columns = preprocess(df)
    model, X_train = train_xgb(X, y)
    sample = X_train.sample(min(200, len(X_train)), random_state=42)
    save_dep_artifacts(model, ohe, label_encoder, feature_columns, sample)
    return model, ohe, label_encoder, feature_columns, sample

CSV_PATH = "anxiety_depression_data.csv"
SHAP_PNG = "shap_waterfall.png"
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
    "Gender": ["Male", "Female", "Other", "Non-Binary"],
    "Family_History_Mental_Illness": ["Yes", "No"],
    "Substance_Use": ["None", "Occassional", "Frequent"],
}

def df_db_to_depression_df(df_db: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DB columns (snake_case) to ML column names (Title_Case)
    to match what preprocess() expects.
    """
    rename_map = {
        "age": "Age",
        "gender": "Gender",
        "education_level": "Education_Level",
        "employment_status": "Employment_Status",
        "sleep_hours": "Sleep_Hours",
        "physical_activity_hrs": "Physical_Activity_Hrs",
        "social_support_score": "Social_Support_Score",
        "anxiety_score": "Anxiety_Score",
        "depression_score": "Depression_Score",
        "stress_level": "Stress_Level",
        "family_history_mental_illness": "Family_History_Mental_Illness",
        "chronic_illnesses": "Chronic_Illnesses",
        "medication_use": "Medication_Use",
        "therapy": "Therapy",
        "meditation": "Meditation",
        "substance_use": "Substance_Use",
        "financial_stress": "Financial_Stress",
        "work_stress": "Work_Stress",
        "self_esteem_score": "Self_Esteem_Score",
        "life_satisfaction_score": "Life_Satisfaction_Score",
        "loneliness_score": "Loneliness_Score",
    }
    df = df_db.rename(columns=rename_map)
    return df

# ----------------------------
# Helper: class mapping
# ----------------------------
def depression_class(score):
    if score <= 7:
        return "Low"
    if score <= 14:
        return "Medium"
    return "High"


# ----------------------------
# Load dataset
# ----------------------------
def load_data(path=CSV_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    print("✅ Loaded dataset:", df.shape)
    return df


# ----------------------------
# Preprocess dataset
# ----------------------------
def preprocess(df):
    df = df.copy()

    # Numeric columns: coerce and fill median, clip to plausible range
    for col, (lo, hi) in NUMERIC_COLS.items():
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median = df[col].median() if df[col].notna().any() else (lo + hi) / 2.0
        df[col] = df[col].fillna(median).clip(lo, hi)

    # Categorical columns
    for c in CATEGORICAL_COLS.keys():
        if c not in df:
            df[c] = "Missing"
        df[c] = df[c].fillna("Missing").astype(str)

    # Target required
    if "Depression_Score" not in df.columns:
        raise KeyError("Dataset must contain 'Depression_Score' column.")
    df["Class"] = df["Depression_Score"].apply(depression_class)
    label_encoder = LabelEncoder()
    df["y"] = label_encoder.fit_transform(df["Class"])

    # One-hot encode categoricals (handle_unknown=ignore)
    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

    cat_cols = list(CATEGORICAL_COLS.keys())
    ohe_arr = ohe.fit_transform(df[cat_cols])
    ohe_cols = ohe.get_feature_names_out(cat_cols)
    ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df.index)

    # Final X and y
    X = pd.concat([df[list(NUMERIC_COLS.keys())].reset_index(drop=True),
                   ohe_df.reset_index(drop=True)], axis=1)
    y = df["y"].astype(int).reset_index(drop=True)

    return X, y, ohe, label_encoder, list(X.columns)


# ----------------------------
# Train model
# ----------------------------
def train_xgb(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
        use_label_encoder=False,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    print(f"\n🎯 Model trained — holdout accuracy: {acc:.3f}\n")
    print(classification_report(y_te, preds, zero_division=0))
    return model, X_tr


# ----------------------------
# Build user feature vector
# ----------------------------
def build_user_features(user_input, ohe, feature_columns):
    user_df = pd.DataFrame([user_input])

    # Ensure categorical columns exist
    for c in CATEGORICAL_COLS.keys():
        if c not in user_df:
            user_df[c] = "Missing"

    # Transform OHE safely
    try:
        ohe_arr = ohe.transform(user_df[list(CATEGORICAL_COLS.keys())])
        ohe_cols = ohe.get_feature_names_out(list(CATEGORICAL_COLS.keys()))
    except Exception:
        ohe_cols = ohe.get_feature_names_out(list(CATEGORICAL_COLS.keys()))
        ohe_arr = np.zeros((1, len(ohe_cols)))

    ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols)

    # Ensure numeric present
    for c in NUMERIC_COLS.keys():
        if c not in user_df:
            user_df[c] = float(np.mean(NUMERIC_COLS[c]))

    num_df = user_df[list(NUMERIC_COLS.keys())].astype(float)

    X_user = pd.concat([num_df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)

    # Add missing columns with zero
    for c in feature_columns:
        if c not in X_user.columns:
            X_user[c] = 0.0

    X_user = X_user[feature_columns]
    return X_user


# ----------------------------
# Correct collapse: pick active dummy (RIGHT-split grouping)
# ----------------------------
def collapse_one_hot_feature(sv, names, vals):
    """
    Collapse one-hot encoded groups properly:
      - group by everything before last underscore
      - pick SHAP value of active dummy (value == 1)
      - fallback: pick dummy with largest absolute shap
    """
    sv = np.array(sv, dtype=float)
    names = np.array(names, dtype=str)
    vals = np.array(vals, dtype=float)

    groups = defaultdict(list)
    for i, n in enumerate(names):
        parts = n.split("_")
        if len(parts) > 1:
            base = "_".join(parts[:-1])  # everything except last token
            groups[base].append(i)

    used = set()
    new_sv = []
    new_names = []
    new_vals = []

    # Collapse grouped categories
    for base, idxs in groups.items():
        if len(idxs) <= 1:
            continue
        # find active dummy (value == 1)
        group_vals = vals[idxs]
        active_local = np.where(group_vals == 1)[0]
        if active_local.size > 0:
            chosen_local_idx = active_local[0]
            chosen_idx = idxs[chosen_local_idx]
        else:
            # fallback: choose idx with largest abs shap
            chosen_idx = idxs[np.argmax(np.abs(sv[idxs]))]

        new_sv.append(float(sv[chosen_idx]))
        new_names.append(base)
        new_vals.append(float(vals[chosen_idx]))
        used.update(idxs)

    # Add remaining (non-grouped or singletons)
    for i in range(len(names)):
        if i in used:
            continue
        new_sv.append(float(sv[i]))
        new_names.append(names[i])
        new_vals.append(float(vals[i]) if i < len(vals) else 0.0)

    return np.array(new_sv), np.array(new_names), np.array(new_vals)


# ----------------------------
# SHAP waterfall: robust + save PNG
# ----------------------------
def shap_waterfall_plot(model, user_features, X_train, top_n=TOP_SHAP_N, save_path=SHAP_PNG):
    # Create explainer (prefer booster)
    try:
        explainer = shap.TreeExplainer(model.get_booster(), data=X_train, model_output="raw")
    except Exception:
        try:
            explainer = shap.TreeExplainer(model, data=X_train, model_output="raw")
        except Exception:
            explainer = shap.TreeExplainer(model)

    # Compute SHAP values (classic API then new API)
    values = None
    base_values = None
    try:
        values = explainer.shap_values(user_features)
        base_values = getattr(explainer, "expected_value", None)
    except Exception:
        try:
            out = explainer(user_features)
            if hasattr(out, "values"):
                values = out.values
                base_values = getattr(out, "base_values", None)
        except Exception as e:
            print("⚠ Could not compute SHAP values:", e)
            return []

    # Determine predicted class
    pred_class = 0
    try:
        proba = model.predict_proba(user_features)[0]
        pred_class = int(np.argmax(proba))
    except Exception:
        proba = None

    # Extract 1D shap vector for class
    if isinstance(values, list):
        idx = min(pred_class, len(values) - 1)
        arr = np.array(values[idx])
        if arr.ndim == 2:
            sv = arr[0]
        else:
            sv = arr.flatten()
        try:
            base = float(np.array(base_values).flatten()[idx])
        except Exception:
            base = float(base_values) if np.isscalar(base_values) else 0.0
    else:
        arr = np.array(values)
        if arr.ndim == 2:
            sv = arr[0]
        else:
            sv = arr.flatten()
        if base_values is None:
            try:
                base_values = explainer.expected_value
            except Exception:
                base_values = 0.0
        try:
            base = float(np.array(base_values).flatten()[0]) if not np.isscalar(base_values) else float(base_values)
        except Exception:
            base = 0.0

    names = np.array(user_features.columns)
    vals = user_features.values.flatten()

    # Collapse one-hot groups properly (active dummy)
    try:
        sv, names, vals = collapse_one_hot_feature(sv, names, vals)
    except Exception as e:
        print("⚠ collapse_one_hot_feature failed:", e)

    # Align and sort by absolute impact
    m = min(len(sv), len(names), len(vals))
    sv = sv[:m]; names = names[:m]; vals = vals[:m]
    order = np.argsort(-np.abs(sv))
    sv = sv[order]; names = names[order]; vals = vals[order]

    # Group remaining features into "other" if too many
    if len(sv) > top_n:
        other_sum = float(np.sum(sv[top_n:]))
        sv = np.concatenate([sv[:top_n], [other_sum]])
        names = np.concatenate([names[:top_n], [f"{len(names)-top_n} other features"]])
        vals = np.concatenate([vals[:top_n], [0.0]])

    # Try to build shap.Explanation and plot
    try:
        exp = shap.Explanation(values=sv.astype(float), base_values=float(base), data=vals.astype(float), feature_names=names.astype(str))
        shap.plots.waterfall(exp, max_display=len(sv))
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=140)
        try:
            plt.show()
        except Exception:
            pass
        print(f"✅ SHAP waterfall saved to {save_path}")
    except Exception as e:
        print("⚠ SHAP waterfall plotting failed:", e)
        print("\nTop SHAP contributors (fallback):")
        for n, s in zip(names[:top_n], sv[:top_n]):
            print(f" - {n}: {s:.4f}")

    # return top features for report
    top_pairs = list(zip(names[:min(len(names), top_n)], sv[:min(len(sv), top_n)]))
    return top_pairs


# ----------------------------
# Extract top shap features (wrapper)
# ----------------------------
def extract_top_shap_features(model, user_features, X_train, top_n=5):
    top = shap_waterfall_plot(model, user_features, X_train, top_n=top_n)
    return top or []


# ----------------------------
# Recommendations generator
# ----------------------------
def auto_recommend(user_input):
    recs = []
    if user_input.get("Sleep_Hours", 8) < 6:
        recs.append("Try to increase nightly sleep towards 7–8 hours; keep a consistent bedtime.")
    if user_input.get("Social_Support_Score", 10) < 4:
        recs.append("Reach out to friends/family or join a local group to improve social support.")
    if user_input.get("Stress_Level", 0) > 6:
        recs.append("Practice short relaxation routines (deep breathing, brief walks, 10 min mindfulness).")
    if user_input.get("Self_Esteem_Score", 10) < 4:
        recs.append("Try journaling and small goals to build self-confidence.")
    if not recs:
        recs.append("Your reported indicators look relatively balanced — maintain healthy routines.")
    return recs


# ----------------------------
# PHQ-9 interactive screening
# ----------------------------
PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure",
    "Trouble concentrating on things",
    "Moving or speaking slowly; or being fidgety/restless",
    "Thoughts that you would be better off dead or hurting yourself"
]


def prompt_phq9():
    print("\n🧾 PHQ-9 (0 = Not at all ... 3 = Nearly every day)")
    answers = []
    for q in PHQ9_QUESTIONS:
        while True:
            try:
                v = int(input(f"{q} (0-3): "))
                if 0 <= v <= 3:
                    answers.append(v)
                    break
            except Exception:
                pass
            print("Enter integer 0..3.")
    total = sum(answers)
    if total <= 4:
        level = "None–Minimal"
    elif total <= 9:
        level = "Mild"
    elif total <= 14:
        level = "Moderate"
    elif total <= 19:
        level = "Moderately Severe"
    else:
        level = "Severe"
    print(f"PHQ-9 total: {total}/27 → {level}")
    return total, level, answers


# ----------------------------
# Human-readable report generation
# ----------------------------

def generate_human_report(pred_label, phq_level, user_input, shap_top, recs):
    """
    Generates a long, descriptive, natural-language psychological report.
    NOTE: Ye sirf LONG TEXT return karega.
    Summarization ab app.py ke andar hogi (stress jaisa).
    """

    # 1. Lifestyle description
    lifestyle_text = (
        f"You are {user_input.get('Age', 'N/A')} years old. "
        f"You sleep around {user_input.get('Sleep_Hours', 'N/A')} hours per night, "
        f"which can influence your mood stability and emotional resilience. "
        f"Your social support score is {user_input.get('Social_Support_Score', 'N/A')}/10, "
        f"indicating the level of emotional and interpersonal help you feel you have. "
        f"Your self-reported stress level is {user_input.get('Stress_Level', 'N/A')}/10, "
        f"which plays a strong role in shaping overall mental health. "
    )

    # 2. SHAP explanation
    shap_sentences = []
    for feat, val in shap_top[:5]:
        try:
            v = float(val)
        except:
            continue

        direction = "increases" if v > 0 else "reduces"
        pretty = feat.replace("_", " ").title()

        shap_sentences.append(
            f"The feature '{pretty}' has a SHAP impact value of {abs(v):.2f}, "
            f"indicating that it {direction} your predicted depression severity."
        )

    shap_text = " ".join(shap_sentences) if shap_sentences else (
        "The model did not identify any dominant contributing features."
    )

    # 3. Recommendations narrative
    rec_text = " ".join([
        f"One suggested improvement is: {r}" for r in recs[:3]
    ])

    # 4. LONG INPUT (raw report)
    long_input = (
        f"The machine learning model classified your depression severity as {pred_label}. "
        f"The PHQ-9 clinical screening result shows {phq_level} symptoms. "
        f"{lifestyle_text} "
        f"{shap_text} "
        f"{rec_text} "
        f"Together, these factors create a comprehensive picture of your current emotional "
        f"and psychological state, revealing how daily habits, internal stress, and personal "
        f"background interact to influence your mental health."
    )

    # yahi return karega, summarizer nahi bulaayega
    return long_input