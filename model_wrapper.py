# model_wrapper.py
import os, json, joblib, base64
import numpy as np
import pandas as pd
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shap
import plotly.graph_objects as go

# Directory where model + encoder + feature_columns + X_train_sample are saved
MODEL_DIR = os.getenv("MODEL_DIR", "./saved_models")


# -------------------------------------------------
# 1. Load artifacts
# -------------------------------------------------
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.joblib"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
    target_encoder = joblib.load(os.path.join(MODEL_DIR, "target_encoder.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)
    X_train_sample = joblib.load(os.path.join(MODEL_DIR, "X_train_sample.joblib"))
    return model, encoders, target_encoder, feature_columns, X_train_sample


# -------------------------------------------------
# 2. User dict → model-ready features
# -------------------------------------------------
def build_user_features(user_input, encoders, feature_columns):
    """
    Convert raw user dict → model-ready 1-row DataFrame.
    Handles numeric conversion + OHE to match training.
    """
    user_df = pd.DataFrame([user_input]).copy()

    numeric_fields = [
        "Age", "Sleep_Duration", "Sleep_Quality",
        "Physical_Activity", "Screen_Time",
        "Caffeine_Intake", "Alcohol_Intake",
        "Work_Hours", "Travel_Time", "Social_Interactions",
        "Blood_Pressure", "Cholesterol_Level", "Blood_Sugar_Level",
    ]

    # Force numeric columns
    for col in numeric_fields:
        if col in user_df:
            user_df[col] = pd.to_numeric(user_df[col], errors="coerce").fillna(0)

    # Categorical columns used by OHE
    categorical_cols = encoders.get("ohe_columns", [])
    for col in categorical_cols:
        if col in user_df:
            user_df[col] = user_df[col].astype(str).fillna("")

    # OneHotEncoder
    ohe = encoders["OneHotEncoder"]
    if categorical_cols:
        ohe_data = ohe.transform(user_df[categorical_cols])
        ohe_df = pd.DataFrame(
            ohe_data,
            columns=ohe.get_feature_names_out(categorical_cols),
            index=user_df.index,
        )
    else:
        ohe_df = pd.DataFrame(index=user_df.index)

    # Numerical feature columns: those in feature_columns not created by OHE
    numerical_cols = [
        c for c in feature_columns
        if c not in ohe.get_feature_names_out(categorical_cols)
    ]

    user_features = pd.concat([user_df[numerical_cols], ohe_df], axis=1)

    # Ensure all expected features exist
    for col in feature_columns:
        if col not in user_features.columns:
            user_features[col] = 0

    # Reorder to match training
    return user_features[feature_columns]


# -------------------------------------------------
# 3. Prediction wrapper
# -------------------------------------------------
def predict(model, user_input, encoders, target_encoder, feature_columns):
    """
    Returns dict:
      {
        "label": decoded label,
        "proba": list of class probabilities,
        "user_features": 1-row DataFrame,
        "pred_encoded": integer class index
      }
    """
    user_features = build_user_features(user_input, encoders, feature_columns)
    pred_encoded = int(model.predict(user_features)[0])
    proba = model.predict_proba(user_features)[0].tolist()
    pred_label = target_encoder.inverse_transform([pred_encoded])[0]

    return {
        "label": pred_label,
        "proba": proba,
        "user_features": user_features,
        "pred_encoded": pred_encoded,
    }


# -------------------------------------------------
# 4. Legacy PNG SHAP waterfall (still available)
# -------------------------------------------------
def shap_waterfall_png_base64(model, user_features, X_train_sample, max_display=8):
    """
    Non-interactive PNG SHAP waterfall for backward compatibility.
    """
    explainer = shap.Explainer(model.predict_proba, X_train_sample)
    shap_values = explainer(user_features)

    proba = model.predict_proba(user_features)[0]
    pred_class = int(np.argmax(proba))

    # SHAP values for predicted class
    sv_vec = shap_values.values[0][:, pred_class]

    # Base value = mean class prob over background
    base = np.mean(model.predict_proba(X_train_sample)[:, pred_class])

    exp = shap.Explanation(
        values=sv_vec,
        base_values=base,
        data=user_features.values[0],
        feature_names=user_features.columns,
    )

    plt.figure(figsize=(8, 4))
    shap.plots.waterfall(exp, max_display=max_display, show=False)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.getvalue()).decode("utf-8")


# -------------------------------------------------
# 5. Unified interactive SHAP (stress + depression)
# -------------------------------------------------
def shap_waterfall_interactive(model, user_features, X_train_sample, top_k=8):
    explainer = shap.Explainer(model.predict_proba, X_train_sample)
    shap_values = explainer(user_features)

    proba = model.predict_proba(user_features)[0]
    pred_class = int(np.argmax(proba))

    sv_vec = shap_values.values[0][:, pred_class]
    feature_names = list(user_features.columns)
    values = user_features.values[0]

    triples = list(zip(feature_names, values, sv_vec))
    triples_sorted = sorted(triples, key=lambda x: abs(x[2]), reverse=True)[:top_k]

    features = [t[0] for t in triples_sorted]
    user_vals = [float(t[1]) for t in triples_sorted]
    impacts = [float(t[2]) for t in triples_sorted]

    # 🟢 ADD DIRECTION HERE
    customdata = [
        [
            f,
            v,
            s,
            "pushes towards predicted class" if s > 0 else "pushes away from predicted class"
        ]
        for f, v, s in zip(features, user_vals, impacts)
    ]

    colors = ["#ff4d6d" if s > 0 else "#4169e1" for s in impacts]

    fig = go.Figure(
        data=[
            go.Bar(
                x=impacts,
                y=features,
                orientation="h",
                marker=dict(color=colors, line=dict(color="#222", width=1)),
                customdata=customdata,
                hovertemplate=(
                    "Your <b>%{customdata[0]}</b> is <b>%{customdata[1]}</b>, "
                    "which <b>%{customdata[3]}</b>.<br>"
                    "SHAP impact: %{customdata[2]:.3f}"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title="<b>Top Factors Affecting Your Prediction</b>",
        xaxis_title="SHAP Value (impact)",
        yaxis_title="Feature",
        template="plotly_white",
        font=dict(size=15),
        height=500,
        bargap=0.25,
        margin=dict(l=130, r=40, t=70, b=40),
        yaxis=dict(autorange="reversed")
    )

    return fig.to_json()

# -------------------------------------------------
# 6. Top features for text explanation
# -------------------------------------------------
def get_shap_top_features(model, user_features, X_train_sample, max_display=8):
    """
    Returns list of (feature_name, shap_value) sorted by |impact|.
    Useful for generating textual explanation.
    """
    explainer = shap.Explainer(model.predict_proba, X_train_sample)
    shap_values = explainer(user_features)

    proba = model.predict_proba(user_features)[0]
    pred_class = int(np.argmax(proba))

    sv = shap_values.values[0][:, pred_class]

    feature_list = list(zip(user_features.columns, sv))
    feature_list_sorted = sorted(feature_list, key=lambda x: abs(x[1]), reverse=True)

    return feature_list_sorted[:max_display]