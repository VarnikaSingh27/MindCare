import pandas as pd
from db_init import db, app
from models import DepressionData

df = pd.read_csv("anxiety_depression_data.csv")

with app.app_context():
    for _, row in df.iterrows():
        record = DepressionData(
            age = row["Age"],
            gender = row["Gender"],
            education_level = row["Education_Level"],
            employment_status = row["Employment_Status"],
            sleep_hours = row["Sleep_Hours"],
            physical_activity_hrs = row["Physical_Activity_Hrs"],
            social_support_score = row["Social_Support_Score"],
            anxiety_score = row["Anxiety_Score"],
            depression_score = row["Depression_Score"],
            stress_level = row["Stress_Level"],
            family_history_mental_illness = row["Family_History_Mental_Illness"],
            chronic_illnesses = row["Chronic_Illnesses"],
            medication_use = str(row["Medication_Use"]),
            therapy = str(row["Therapy"]),
            meditation = str(row["Meditation"]),
            substance_use = str(row["Substance_Use"]),
            financial_stress = row["Financial_Stress"],
            work_stress = row["Work_Stress"],
            self_esteem_score = row["Self_Esteem_Score"],
            life_satisfaction_score = row["Life_Satisfaction_Score"],
            loneliness_score = row["Loneliness_Score"]
        )
        db.session.add(record)

    db.session.commit()

print("Depression data imported successfully")