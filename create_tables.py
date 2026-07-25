from app import app          # import Flask app from app.py
from db_init import db
from models import (
    HealthData, DepressionData, AnxietyData,
    User, Assessment, AssessmentHistory
)

with app.app_context():
    db.create_all()
    print("All tables created successfully!")