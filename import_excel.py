import pandas as pd
from db_init import db, app
from models import HealthData

# Replace with your actual Excel file path
excel_file = "stress_detection_data (1).csv"  

# Read Excel file into a DataFrame
df = pd.read_csv(excel_file)

# Insert data into the database
with app.app_context():
    for _, row in df.iterrows():
        health_data = HealthData(
            age=row.get('Age'),
            gender=row.get('Gender'),
            occupation=row.get('Occupation'),
            marital_status=row.get('Marital_Status'),
            sleep_duration=row.get('Sleep_Duration'),
            sleep_quality=row.get('Sleep_Quality'),
            physical_activity=row.get('Physical_Activity'),
            screen_time=row.get('Screen_Time'),
            caffeine_intake=row.get('Caffeine_Intake'),
            alcohol_intake=row.get('Alcohol_Intake'),
            smoking_habit=row.get('Smoking_Habit'),
            work_hours=row.get('Work_Hours'),
            travel_time=row.get('Travel_Time'),
            social_interactions=row.get('Social_Interactions'),
            meditation_practice=row.get('Meditation_Practice'),
            exercise_type=row.get('Exercise_Type'),
            blood_pressure=row.get('Blood_Pressure'),
            cholesterol_level=row.get('Cholesterol_Level'),
            blood_sugar_level=row.get('Blood_Sugar_Level'),
            stress_detection=row.get('Stress_Detection')
        )
        db.session.add(health_data)

    db.session.commit()
    print("Excel data imported successfully!")
