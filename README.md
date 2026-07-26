# MindCare  — Your Personal Mental Health Companion

MindCare is a full-stack web application that uses machine learning to help users understand and track their **stress**, **anxiety**, and **depression** risk. It combines validated psychological screening tools (PSS-10, GAD-7, PHQ-9) with XGBoost models trained on lifestyle and physiological data, then explains every prediction with SHAP so users can see *why* the model reached its conclusion — not just the result.

## Screenshots
<img width="1280" height="463" alt="WhatsApp Image 2026-07-26 at 10 48 28 AM" src="https://github.com/user-attachments/assets/e6d7905f-f7aa-4941-9cb7-0a606c26d3bb" />
<br>
<img width="1280" height="582" alt="WhatsApp Image 2026-07-26 at 10 48 27 AM" src="https://github.com/user-attachments/assets/6c0981b2-960c-415f-b76c-4eceb326c907" />
<br>
<img width="1600" height="747" alt="WhatsApp Image 2026-07-26 at 10 48 30 AM" src="https://github.com/user-attachments/assets/f49a4de4-15bc-4902-9191-b20eddde0139" />
<br>
<img width="1518" height="555" alt="WhatsApp Image 2026-07-26 at 10 48 31 AM" src="https://github.com/user-attachments/assets/1171bfce-cade-46d5-b7d1-2b5c51015c1a" />
<br>
<img width="1252" height="722" alt="WhatsApp Image 2026-07-26 at 10 48 31 AM (1)" src="https://github.com/user-attachments/assets/81dbd3e0-78b5-44b4-86a3-c1d3f435bb68" />

---

##  Features

- **Three independent risk assessments**
  - **Stress** — lifestyle/physiological inputs + the **PSS-10** (Perceived Stress Scale) questionnaire
  - **Anxiety** — lifestyle/physiological inputs + the **GAD-7** questionnaire
  - **Depression** — lifestyle/physiological inputs + the **PHQ-9** questionnaire
- **ML-powered predictions** using trained **XGBoost** classifiers for each condition
- **Explainable AI** — SHAP waterfall charts (interactive JSON + static PNG) show exactly which factors influenced each prediction and by how much
- **Plain-language summary reports**, auto-generated and condensed with a Hugging Face summarization model (`facebook/bart-large-cnn`)
- **Personalized recommendations** based on sleep, activity, screen time, caffeine/alcohol intake, and other inputs
- **User accounts** — signup/login with hashed passwords
- **Assessment history** — every submission is logged so users can track trends over time
- **Interactive dashboard** — mood tracker, quick notes, wellness tips, progress overview, flashcards, and a guided breathing/relax mode
- **Profile & settings management**

---

## Tech Stack

**Backend**
- [Flask](https://flask.palletsprojects.com/) (REST API)
- [Flask-SQLAlchemy](https://flask-sqlalchemy.palletsprojects.com/) + PostgreSQL
- [Flask-CORS](https://flask-cors.readthedocs.io/)
- [XGBoost](https://xgboost.readthedocs.io/) for classification models
- [SHAP](https://shap.readthedocs.io/) for model explainability
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) for report summarization
- [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) for data processing

**Frontend**
- HTML5, CSS3, and vanilla JavaScript (served as static files by Flask)

**Data / Models**
- Custom datasets for stress, anxiety, and depression (CSV)
- Pre-trained model artifacts stored under `saved_models/` and `saved_models_depression/`

---

##  Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL (running locally or remotely)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/MindCare.git
cd MindCare
```

### 2. Create a virtual environment & install dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://<user>:<password>@localhost:5432/healthdb
MODEL_DIR=./saved_models
FLASK_ENV=development
PORT=5000
```

### 4. Set up the database

```bash
python create_tables.py
```

Then import the training datasets:

```bash
python import_excel.py
python import_anxiety.py
python import_depression.py
```

### 5. Run the app

```bash
python app.py
```

On startup, the app automatically trains and loads the stress, anxiety, and depression models from the data in your database. Once ready, open: http://localhost:5000
