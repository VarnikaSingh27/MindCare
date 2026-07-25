import pandas as pd
from db_init import db, app
from models import AnxietyData

df = pd.read_csv("anxiety_data.csv")

with app.app_context():
    for _, row in df.iterrows():
        entry = AnxietyData(
            age = row["Age"],
            gender = row["Gender"],
            occupation = row["Occupation"],
            sleep_hours = row["Sleep Hours"],
            physical_activity = row["Physical Activity (hrs/week)"],
            caffeine_intake = row["Caffeine Intake (mg/day)"],
            alcohol_consumption = row["Alcohol Consumption (drinks/week)"],
            smoking = row["Smoking"],
            family_history_anxiety = row["Family History of Anxiety"],
            stress_level = row["Stress Level (1-10)"],
            heart_rate = row["Heart Rate (bpm)"],
            breathing_rate = row["Breathing Rate (breaths/min)"],
            sweating_level = row["Sweating Level (1-5)"],
            dizziness = row["Dizziness"],
            medication = row["Medication"],
            therapy_sessions = row["Therapy Sessions (per month)"],
            recent_life_event = row["Recent Major Life Event"],
            diet_quality = row["Diet Quality (1-10)"],
            anxiety_level = row["Anxiety Level (1-10)"]
        )

        db.session.add(entry)

    db.session.commit()
    print("Anxiety CSV imported successfully!")