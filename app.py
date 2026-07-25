# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import traceback
import joblib
import numpy as np
from flask import Flask
from db_init import db
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:priya2004@localhost:5432/healthdb"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
CORS(app)
load_dotenv()

from db_utils import read_table_as_df
from preprocess import df_db_to_model
from model_training import preprocess_features, train_xgb, save_artifacts
from model_wrapper import load_artifacts, predict, shap_waterfall_png_base64, shap_waterfall_interactive, build_user_features, get_shap_top_features
from pss_utils import compute_pss_level
from recommendations import auto_recommend
from flask import Flask, request, jsonify
from depression_model_predict import predict_depression
from models import DepressionData, db
from werkzeug.security import generate_password_hash, check_password_hash
from models import User, Assessment, AssessmentHistory

from depression_model import (
    train_depression_from_df,
    load_dep_artifacts,
    build_user_features as build_dep_user_features,
    auto_recommend as dep_auto_recommend,
    generate_human_report as dep_generate_report,
    shap_waterfall_plot,   # if you want SHAP PNG too
)

from anxiety_model import (
    train_anxiety_from_df,
    predict_anxiety_level_xgb,
    get_shap_top_features,
    shap_waterfall_png_base64,
    auto_recommend as anx_auto_recommend,
    generate_subjective_report as anx_generate_report,
)

ANX_MODEL = None
ANX_ENCODERS = {}
ANX_TARGET_ENCODER = None
ANX_FEATURE_COLUMNS = None
ANX_X_TRAIN_SAMPLE = None

DEP_MODEL = None
DEP_OHE = None
DEP_LABEL_ENCODER = None
DEP_FEATURE_COLUMNS = None
DEP_X_TRAIN_SAMPLE = None

MODEL_DIR = os.getenv("MODEL_DIR", "./saved_models")
TABLE_NAME = "health_data"

MODEL = None
ENCODERS = None
TARGET_ENCODER = None
FEATURE_COLUMNS = None
X_TRAIN_SAMPLE = None

def train_depression_on_startup():
    global DEP_MODEL, DEP_OHE, DEP_LABEL_ENCODER, DEP_FEATURE_COLUMNS, DEP_X_TRAIN_SAMPLE
    print("🔄 Training depression model from DB...")
    df_dep = read_table_as_df("depression_data")
    if df_dep.shape[0] == 0:
        print("⚠ No rows found in depression_data table.")
        return
    DEP_MODEL, DEP_OHE, DEP_LABEL_ENCODER, DEP_FEATURE_COLUMNS, DEP_X_TRAIN_SAMPLE = train_depression_from_df(df_dep)
    print("✅ Depression model trained & loaded")

def train_on_startup():
    global MODEL, ENCODERS, TARGET_ENCODER, FEATURE_COLUMNS, X_TRAIN_SAMPLE
    print("🔄 Loading data from DB and training model (startup)...")
    df_raw = read_table_as_df(TABLE_NAME)
    if df_raw.shape[0] == 0:
        raise RuntimeError("No rows found in health_data table.")
    df = df_db_to_model(df_raw)
    X, y, encoders, target_encoder, feature_columns = preprocess_features(df)
    model, X_train = train_xgb(X, y, target_encoder)
    # save artifacts (sample X_train)
    sample = X_train.sample(min(200, len(X_train)), random_state=42)
    save_artifacts(model, encoders, target_encoder, feature_columns, sample)
    print("✅ Saved model artifacts.")
    MODEL, ENCODERS, TARGET_ENCODER, FEATURE_COLUMNS, X_TRAIN_SAMPLE = load_artifacts()
    print("✅ Model loaded into memory.")

def train_anxiety_on_startup():
    global ANX_MODEL, ANX_ENCODERS, ANX_TARGET_ENCODER, ANX_FEATURE_COLUMNS, ANX_X_TRAIN_SAMPLE
    print("🔄 Training anxiety model from DB...")

    df_anx = read_table_as_df("anxiety_data")

    if df_anx.shape[0] == 0:
        print("⚠ No rows found in anxiety_data table.")
        return

    (ANX_MODEL,
     ANX_ENCODERS,
     ANX_TARGET_ENCODER,
     ANX_FEATURE_COLUMNS,
     ANX_X_TRAIN_SAMPLE) = train_anxiety_from_df(df_anx)

    print("✅ Anxiety model trained & loaded")

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.json

    # ---- FIX: Convert types ----
    numeric_fields = [
        "Age", "Sleep_Duration", "Sleep_Quality", "Physical_Activity",
        "Screen_Time", "Caffeine_Intake", "Alcohol_Intake",
        "Work_Hours", "Travel_Time", "Social_Interactions",
        "Blood_Pressure", "Cholesterol_Level", "Blood_Sugar_Level"
    ]

    for f in numeric_fields:
        if f in data:
            data[f] = float(data[f])

    # integer fields
    int_fields = ["Age", "Sleep_Quality"]
    for f in int_fields:
        data[f] = int(data[f])

    # yes/no normalization
    if "Smoking_Habit" in data:
        data["Smoking_Habit"] = "Yes" if data["Smoking_Habit"] == "Yes" else "No"

    if "Meditation_Practice" in data:
        data["Meditation_Practice"] = "Yes" if data["Meditation_Practice"] == "Yes" else "No"

    # Predict
    pred_label, proba, user_features = predict(
        MODEL,
        data,
        ENCODERS,
        TARGET_ENCODER,
        FEATURE_COLUMNS
    )

    return jsonify({
        "prediction": pred_label,
        "probabilities": proba.tolist()
    })

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    Expects payload:
    {
      "ml_features": { Age, Gender, ... (18 mapped fields) },
      "pss_answers": [10 ints 0-4]
    }
    Returns prediction, probabilities, pss score & level, recommendations, summary, shap image base64.
    """
    global MODEL, ENCODERS, TARGET_ENCODER, FEATURE_COLUMNS, X_TRAIN_SAMPLE
    if MODEL is None:
        return jsonify({"error": "Model not ready (still training?)"}), 503
    payload = request.get_json()
    if not payload:
        return jsonify({"error":"No JSON body received"}), 400
    ml_features = payload.get("ml_features")
    pss_answers = payload.get("pss_answers")
    if ml_features is None or pss_answers is None:
        return jsonify({"error":"Both 'ml_features' and 'pss_answers' required"}), 400
    try:
        # Ensure data types: we trust preprocess/build_user_features to handle missing keys by setting defaults
        # 1) Predict
        pred_res = predict(MODEL, ml_features, ENCODERS, TARGET_ENCODER, FEATURE_COLUMNS)
        user_features = pred_res['user_features']
        # 2) SHAP – INTERACTIVE JSON
        shap_interactive = shap_waterfall_interactive(MODEL, user_features, X_TRAIN_SAMPLE)
        # 3) SHAP top features
        shap_top = get_shap_top_features(MODEL, user_features, X_TRAIN_SAMPLE)[:8]  # top 8
        # 4) PSS scoring
        pss_score, pss_level, pss_rev = compute_pss_level(pss_answers)
        # 5) Recommendations based on ml_features (we will coerce numeric-like fields to numbers when possible)
        # Ensure numeric keys exist in ml_features for recommendation checks
        rec_input = {}
        for k in ['Sleep_Duration','Sleep_Quality','Physical_Activity','Screen_Time','Caffeine_Intake','Alcohol_Intake','Blood_Pressure','Cholesterol_Level','Blood_Sugar_Level']:
            rec_input[k] = ml_features.get(k)
            try:
                # attempt numeric conversion
                if rec_input[k] is not None:
                    rec_input[k] = float(rec_input[k])
            except Exception:
                rec_input[k] = 0
        recs = auto_recommend(rec_input)
        # 6) Generate summary report using template + summarizer if available
        # Build long_text similar to your generate_subjective_report
        shap_text_parts = []
        for feat, val in shap_top[:5]:
            direction = "increases" if float(val) > 0 else "reduces"
            shap_text_parts.append(f"The feature '{feat}' has a SHAP impact score of {abs(float(val)):.2f}, which means it significantly {direction} your predicted stress level.")
        shap_text = " ".join(shap_text_parts)
        lifestyle_text = (
            f"Your lifestyle data shows that you sleep for {ml_features.get('Sleep_Duration')} hours "
            f"with a sleep quality of {ml_features.get('Sleep_Quality')}. "
            f"You engage in physical activity {ml_features.get('Physical_Activity')} days per week "
            f"and spend about {ml_features.get('Screen_Time')} hours per day on screens. "
            f"You consume {ml_features.get('Caffeine_Intake')} cups of caffeine per day "
            f"and drink alcohol {ml_features.get('Alcohol_Intake')} times per week."
        )
        long_text = (
            f"The machine learning model predicted your stress level as {pred_res['label']}. "
            f"The PSS-10 psychological test indicates a {pss_level} stress level (score {pss_score}). "
            f"{lifestyle_text} {shap_text} "
            f"In summary, recommended improvements include: {'; '.join(recs[:3])}."
        )
        summary = None
        try:
            # lazy import summarizer to avoid heavy import at top
            from transformers import pipeline
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            out = summarizer(long_text, max_length=180, min_length=120, do_sample=False)[0]
            summary = out.get("summary_text")
        except Exception:
            # fallback to raw long_text if transformers unavailable
            summary = long_text
        try:
            user_id = payload.get("user_id")
            user = User.query.get(user_id)

            if user:
                # Get most recent assessment (correct)
                latest = Assessment.query.filter_by(user_id=user.user_id).order_by(Assessment.id.desc()).first()

                # If no previous assessment, create new row
                if not latest:
                    latest = Assessment(user_id=user.user_id)
                    db.session.add(latest)

                # Update stress fields
                latest.stress_prediction_ml = pred_res['label']
                latest.stress_score_pss = pss_score
                latest.last_assessment_taken_dateandtime = datetime.utcnow()

                # Add history row
                history = AssessmentHistory(
                    user_id=user.user_id,
                    stress_prediction_ml=pred_res['label'],
                    stress_score_pss=pss_score
                )
                db.session.add(history)

                db.session.commit()

        except Exception as db_error:
            print("⚠ DB SAVE ERROR:", db_error)
        # final response
        response = {
            "prediction": pred_res['label'],
            "probabilities": pred_res['proba'],
            "pss_score": pss_score,
            "pss_level": pss_level,
            "recommendations": recs,
            "summary_report": summary,
            "shap_interactive": shap_interactive,
            "shap_top_features": [{"feature": f, "value": float(v)} for f,v in shap_top[:8]]
        }
        return jsonify(response)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/depression/analyze", methods=["POST"])
def api_depression_analyze():
    global DEP_MODEL, DEP_OHE, DEP_LABEL_ENCODER, DEP_FEATURE_COLUMNS, DEP_X_TRAIN_SAMPLE
    if DEP_MODEL is None:
        return jsonify({"error": "Depression model not ready"}), 503

    payload = request.get_json()
    if not payload:
        return jsonify({"error": "No JSON body received"}), 400

    # ⭐ NEW — GET USER ID
    user_id = payload.get("user_id")

    ml_features = payload.get("ml_features")
    phq_answers = payload.get("phq_answers")

    if ml_features is None or phq_answers is None:
        return jsonify({"error": "Both ml_features and phq_answers required"}), 400

    try:
        # -----------------------------
        # PHQ-9 SCORING
        # -----------------------------
        phq_answers = [int(x) for x in phq_answers]
        phq_total = sum(phq_answers)

        if phq_total <= 4:
            phq_level = "None–Minimal"
        elif phq_total <= 9:
            phq_level = "Mild"
        elif phq_total <= 14:
            phq_level = "Moderate"
        elif phq_total <= 19:
            phq_level = "Moderately Severe"
        else:
            phq_level = "Severe"

        # -----------------------------
        # ML PREDICTION
        # -----------------------------
        X_user = build_dep_user_features(ml_features, DEP_OHE, DEP_FEATURE_COLUMNS)
        proba = DEP_MODEL.predict_proba(X_user)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = DEP_LABEL_ENCODER.inverse_transform([pred_idx])[0]

        # -----------------------------
        # SHAP INTERACTIVE
        # -----------------------------
        shap_interactive = shap_waterfall_interactive(
            DEP_MODEL, X_user, DEP_X_TRAIN_SAMPLE
        )

        # Top features list
        shap_top = get_shap_top_features(DEP_MODEL, X_user, DEP_X_TRAIN_SAMPLE)[:8]

        # -----------------------------
        # RECOMMENDATIONS
        # -----------------------------
        recs = dep_auto_recommend(ml_features)

        # -----------------------------
        # LONG REPORT → summary
        # -----------------------------
        long_report = dep_generate_report(
            pred_label, phq_level, ml_features, shap_top, recs
        )

        try:
            from transformers import pipeline
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            out = summarizer(long_report, max_length=180, min_length=120, do_sample=False)[0]
            summary = out.get("summary_text", long_report)
        except Exception:
            summary = long_report
        # INSIDE api_depression_analyze ROUTE
        try:
            user_id = payload.get("user_id")
            if user_id:
                # make sure user exists
                user = User.query.get(user_id)

                if user:
                    # Add NEW row for depression test
                    history = AssessmentHistory(
                        user_id=user_id,
                        depression_prediction_ml=pred_label,
                        depression_score_phq=phq_total,
                        assessment_taken_at=datetime.utcnow()
                    )
                    db.session.add(history)
                    db.session.commit()

        except Exception as db_error:
            print("⚠️ Depression Save Error:", db_error)

        # -----------------------------
        # RETURN JSON (⭐ includes user_id)
        # -----------------------------
        return jsonify({
            "prediction": pred_label,
            "probabilities": [float(x) for x in proba],
            "phq_total": phq_total,
            "phq_level": phq_level,
            "recommendations": recs,
            "summary_report": summary,
            "shap_interactive": shap_interactive,
            "shap_top_features": [
                {"feature": f, "value": float(v)} for f, v in shap_top
            ],
            "user_id": user_id    # ⭐ IMPORTANT
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/anxiety/analyze", methods=["POST"])
def api_anxiety_analyze():
    """
    Expects:
    {
      "ml_features": { ... anxiety features ... },
      "gad_answers": [7 ints 0-3]
    }
    Returns:
      prediction, probabilities, gad_total/level, summary, recs, shap image, shap_top_features
    """
    global ANX_MODEL, ANX_ENCODERS, ANX_TARGET_ENCODER, ANX_FEATURE_COLUMNS, ANX_X_TRAIN_SAMPLE

    if ANX_MODEL is None:
        return jsonify({"error": "Anxiety model not ready"}), 503

    payload = request.get_json()
    if not payload:
        return jsonify({"error": "No JSON body received"}), 400

    ml_features = payload.get("ml_features")
    gad_answers = payload.get("gad_answers")

    if ml_features is None or gad_answers is None:
        return jsonify({"error": "Both ml_features and gad_answers required"}), 400

    try:
        # -----------------------------
        # GAD-7 scoring
        # -----------------------------
        gad_answers = [int(x) for x in gad_answers]
        gad_total = sum(gad_answers)

        if gad_total <= 4:
            gad_level = "Minimal"
        elif gad_total <= 9:
            gad_level = "Mild"
        elif gad_total <= 14:
            gad_level = "Moderate"
        else:
            gad_level = "Severe"

        # -----------------------------
        # ML prediction (XGB)
        # -----------------------------
        # ml_features me keys ideally wohi hone chahiye
        # jo anxiety dataset me the (Age, Gender, Sleep Hours, ...).
        pred_label, proba, X_user = predict_anxiety_level_xgb(
            ANX_MODEL,
            ml_features,
            ANX_ENCODERS,
            ANX_TARGET_ENCODER,
            ANX_FEATURE_COLUMNS,
        )

        # -----------------------------
        # SHAP waterfall → Base64 PNG
        # -----------------------------
        shap_b64 = shap_waterfall_png_base64(ANX_MODEL, X_user, ANX_X_TRAIN_SAMPLE)

        # Top features list
        shap_top = get_shap_top_features(ANX_MODEL, X_user, ANX_X_TRAIN_SAMPLE, top_k=8)

        # -----------------------------
        # Recommendations
        # -----------------------------
        recs = anx_auto_recommend(ml_features)

        # -----------------------------
        # Long psychological report + summarizer (inside function)
        # -----------------------------
        summary = anx_generate_report(
            pred_label,
            gad_level,
            ml_features,
            shap_top,
            recs,
        )

        # -----------------------------
        # Final response (same pattern as stress/depression)
        # -----------------------------
        return jsonify({
            "prediction": pred_label,
            "probabilities": [float(x) for x in proba],
            "gad_total": gad_total,
            "gad_level": gad_level,
            "recommendations": recs,
            "summary_report": summary,
            "shap_image_base64": shap_b64,
            "shap_top_features": [
                {"feature": f, "value": float(v)} for f, v in shap_top
            ],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        # Check if user already exists
        existing = User.query.filter_by(email=email).first()
        if existing:
            return jsonify({"error": "Email already registered"}), 409

        # Create user
        hashed = generate_password_hash(password)
        user = User(email=email, password=hashed)

        db.session.add(user)
        db.session.commit()

        return jsonify({"message": "Account created successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        user = User.query.filter_by(email=email).first()

        if not user:
            return jsonify({"error": "User not found"}), 404

        if not check_password_hash(user.password, password):
            return jsonify({"error": "Incorrect password"}), 401

        return jsonify({
            "message": "Login successful",
            "user_id": user.user_id,
            "email": user.email
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_assessments/<int:user_id>", methods=["GET"])
def get_assessments(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Latest Assessment
    latest = Assessment.query.filter_by(user_id=user_id).first()
    latest_data = None
    if latest:
        latest_data = {
            "stress_prediction_ml": latest.stress_prediction_ml,
            "stress_score_pss": latest.stress_score_pss,
            "anxiety_prediction_ml": latest.anxiety_prediction_ml,
            "anxiety_score_gad": latest.anxiety_score_gad,
            "depression_prediction_ml": latest.depression_prediction_ml,
            "depression_score_phq": latest.depression_score_phq,
            "timestamp": latest.last_assessment_taken_dateandtime
        }

    # History
    history = AssessmentHistory.query.filter_by(user_id=user_id).order_by(
        AssessmentHistory.assessment_taken_at.desc()
    ).all()

    history_list = [
        {
            "stress_prediction_ml": h.stress_prediction_ml,
            "stress_score_pss": h.stress_score_pss,
            "anxiety_prediction_ml": h.anxiety_prediction_ml,
            "anxiety_score_gad": h.anxiety_score_gad,
            "depression_prediction_ml": h.depression_prediction_ml,
            "depression_score_phq": h.depression_score_phq,
            "timestamp": h.assessment_taken_at.isoformat()
        } for h in history
    ]

    return jsonify({
        "latest": latest_data,
        "history": history_list
    })

@app.route("/get_assessments")
def get_assessments_by_email():
    email = request.args.get("email")
    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({"history": []})

    history = AssessmentHistory.query.filter_by(user_id=user.user_id).order_by(AssessmentHistory.id.asc()).all()

    data = []
    for h in history:
        data.append({
            "date": h.assessment_taken_at.strftime("%Y-%m-%d"),
            "stress_score": h.stress_score_pss,
            "anxiety_score": h.anxiety_score_gad,
            "depression_score": h.depression_score_phq
        })

    return jsonify({"history": data})

if __name__ == "__main__":
    try:
        train_on_startup()            # existing stress training
        train_depression_on_startup() # NEW
        train_anxiety_on_startup()
    except Exception as e:
        print("Training on startup failed (see error). You can still start app but some models won't be available.")
        traceback.print_exc()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)