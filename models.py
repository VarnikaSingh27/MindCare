# models.py
from db_init import db
from datetime import datetime

class HealthData(db.Model):
    __tablename__ = 'health_data'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    occupation = db.Column(db.String(50))
    marital_status = db.Column(db.String(20))
    sleep_duration = db.Column(db.Float)
    sleep_quality = db.Column(db.Integer)
    physical_activity = db.Column(db.Float)
    screen_time = db.Column(db.Float)
    caffeine_intake = db.Column(db.String(20))
    alcohol_intake = db.Column(db.String(20))
    smoking_habit = db.Column(db.String(20))
    work_hours = db.Column(db.Float)
    travel_time = db.Column(db.Float)
    social_interactions = db.Column(db.Float)
    meditation_practice = db.Column(db.String(20))
    exercise_type = db.Column(db.String(50))
    blood_pressure = db.Column(db.String(20))
    cholesterol_level = db.Column(db.String(20))
    blood_sugar_level = db.Column(db.String(20))
    stress_detection = db.Column(db.String(20))
    
    def __repr__(self):
        return f"<HealthData {self.id} - Stress: {self.stress_detection}>"
    
class DepressionData(db.Model):
    __tablename__ = "depression_data"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    education_level = db.Column(db.String(50))
    employment_status = db.Column(db.String(50))

    sleep_hours = db.Column(db.Float)
    physical_activity_hrs = db.Column(db.Float)
    social_support_score = db.Column(db.Integer)
    anxiety_score = db.Column(db.Integer)
    depression_score = db.Column(db.Integer)
    stress_level = db.Column(db.Integer)

    family_history_mental_illness = db.Column(db.Integer)
    chronic_illnesses = db.Column(db.Integer)

    medication_use = db.Column(db.String(50))      # FIXED
    therapy = db.Column(db.Integer)
    meditation = db.Column(db.Integer)
    substance_use = db.Column(db.String(50))       # FIXED

    financial_stress = db.Column(db.Integer)
    work_stress = db.Column(db.Integer)
    self_esteem_score = db.Column(db.Integer)
    life_satisfaction_score = db.Column(db.Integer)
    loneliness_score = db.Column(db.Integer)

    def __repr__(self):
        return f"<DepressionData {self.id} - DepScore: {self.depression_score}>"

class AnxietyData(db.Model):
    __tablename__ = "anxiety_data"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    occupation = db.Column(db.String(100))
    
    sleep_hours = db.Column(db.Float)
    physical_activity = db.Column(db.Float)
    caffeine_intake = db.Column(db.Integer)
    alcohol_consumption = db.Column(db.Integer)
    
    smoking = db.Column(db.String(10))
    family_history_anxiety = db.Column(db.String(10))

    stress_level = db.Column(db.Integer)
    heart_rate = db.Column(db.Integer)
    breathing_rate = db.Column(db.Integer)
    sweating_level = db.Column(db.Integer)

    dizziness = db.Column(db.String(10))
    medication = db.Column(db.String(50))
    therapy_sessions = db.Column(db.Integer)

    recent_life_event = db.Column(db.String(100))
    diet_quality = db.Column(db.Integer)

    anxiety_level = db.Column(db.Integer)

    def __repr__(self):
        return f"<AnxietyData {self.id} - Anxiety: {self.anxiety_level}>"

# ---------------------------------------------------
# 1. USER TABLE (only email + password)
# ---------------------------------------------------
class User(db.Model):
    __tablename__ = "users"

    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)   # hashed password

    def __repr__(self):
        return f"<User {self.user_id} - {self.email}>"


# ---------------------------------------------------
# 2. ASSESSMENT (ONLY LATEST ENTRY PER USER)
# ---------------------------------------------------
class Assessment(db.Model):
    __tablename__ = "assessments"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.user_id"),
        nullable=False
    )

    # Stress
    stress_prediction_ml = db.Column(db.String(20), nullable=True)
    stress_score_pss = db.Column(db.Integer, nullable=True)

    # Anxiety
    anxiety_prediction_ml = db.Column(db.String(20), nullable=True)
    anxiety_score_gad = db.Column(db.Integer, nullable=True)

    # Depression
    depression_prediction_ml = db.Column(db.String(20), nullable=True)
    depression_score_phq = db.Column(db.Integer, nullable=True)

    # Latest timestamp
    last_assessment_taken_dateandtime = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    def __repr__(self):
        return f"<Assessment Latest for User {self.user_id}>"


# ---------------------------------------------------
# 3. ASSESSMENT HISTORY (EVERY SUBMISSION)
# ---------------------------------------------------
class AssessmentHistory(db.Model):
    __tablename__ = "assessment_history"

    history_id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.user_id"),
        nullable=False
    )

    # Stress
    stress_prediction_ml = db.Column(db.String(20), nullable=True)
    stress_score_pss = db.Column(db.Integer, nullable=True)

    # Anxiety
    anxiety_prediction_ml = db.Column(db.String(20), nullable=True)
    anxiety_score_gad = db.Column(db.Integer, nullable=True)

    # Depression
    depression_prediction_ml = db.Column(db.String(20), nullable=True)
    depression_score_phq = db.Column(db.Integer, nullable=True)

    # Timestamp for history log
    assessment_taken_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    def __repr__(self):
        return f"<History {self.history_id} for User {self.user_id}>"