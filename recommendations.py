# recommendations.py
import random

recommendations_dict = {
    "Sleep_Duration_low": [
        "Try maintaining a fixed sleep schedule, even on weekends. Going to bed and waking up at the same time helps your brain regulate the sleep cycle and improves overall rest.",
        "Avoid using mobile phones, laptops, or bright screens at least 1 hour before bed. Blue light suppresses melatonin and makes it harder to fall asleep.",
        "Create a short bedtime routine such as reading, light stretching, warm shower, or calming music. This signals your brain to relax and prepare for sleep."
    ],
    "Sleep_Quality_low": [
        "Practice deep-breathing or mindfulness relaxation before sleeping. This calms the nervous system and improves sleep depth.",
        "Keep your room dark, cool, and quiet. Using blackout curtains or earplugs can improve sleep quality if noise or light is an issue.",
        "Avoid heavy meals or caffeine close to bedtime. Lighter dinners help the body rest better and prevent disturbances at night."
    ],
    "Screen_Time_high": [
        "Take small digital breaks every 45–60 minutes. Look at a distant object or walk around to reduce eye strain and mental fatigue.",
        "Limit late-night screen usage. High screen exposure before sleeping delays melatonin release and causes poor-quality sleep.",
        "Use blue light filters or night mode on devices. This reduces strain on your eyes and lowers stress on the nervous system."
    ],
    "Physical_Activity_low": [
        "Start with easy goals like a 20-minute walk daily. Walking improves blood circulation, reduces stress hormones, and boosts mood.",
        "Include light stretching or yoga in the morning. Gentle movement relaxes muscles and releases tension from the body.",
        "Try simple home workouts 3 times a week. Even basic exercises like squats, planks, and jumping jacks can improve energy levels."
    ],
    "Caffeine_Intake_high": [
        "Avoid caffeine late in the evening. It stays in your system for hours and can disturb your sleep cycle.",
        "Replace some cups of coffee with herbal tea, warm milk, or lemon water. These are calming and better for nighttime relaxation.",
        "Monitor your caffeine intake consciously. Too much caffeine can cause anxiety, restlessness, and poor sleep quality."
    ],
    "Alcohol_Intake_high": [
        "Try limiting alcohol to weekends or special occasions instead of daily consumption. This reduces stress on the liver and improves sleep.",
        "Replace alcoholic drinks with fresh juices, smoothies, or fruit-infused water. It supports hydration and keeps energy stable.",
        "When feeling stressed, avoid alcohol as a coping method. It may give temporary relaxation but increases anxiety later."
    ],
    "Blood_Pressure_high": [
        "Reduce salty and processed foods. Sodium causes water retention and increases pressure on blood vessels.",
        "Practice slow breathing for 5 minutes daily. It activates the relaxation response and lowers stress-related spikes.",
        "Increase hydration and take short walks throughout the day. Light movement improves circulation and reduces pressure on the heart."
    ],
    "Cholesterol_high": [
        "Avoid fried, oily, and fast foods. They contain unhealthy fats that raise cholesterol and increase heart strain.",
        "Add fresh vegetables, fruits, and fiber-rich foods like oats and whole grains. These help clean excess cholesterol naturally.",
        "Stay active with regular walking, cycling, or simple workouts. Physical movement helps balance cholesterol levels."
    ],
    "Blood_Sugar_high": [
        "Avoid sugary drinks, desserts, and packaged snacks. They cause sudden spikes and drops in blood sugar levels.",
        "Have smaller, more frequent meals instead of large heavy ones. This keeps glucose levels more stable throughout the day.",
        "Add fiber-rich foods like vegetables, whole grains, and nuts. Fiber slows sugar absorption and protects blood sugar balance."
    ]
}

def auto_recommend(user):
    recs = []
    try:
        if user.get('Sleep_Duration', 999) < 6:
            recs.append(random.choice(recommendations_dict["Sleep_Duration_low"]))
        if user.get('Sleep_Quality', 999) < 3:
            recs.append(random.choice(recommendations_dict["Sleep_Quality_low"]))
        if user.get('Screen_Time', 0) > 4:
            recs.append(random.choice(recommendations_dict["Screen_Time_high"]))
        if user.get('Physical_Activity', 999) < 2:
            recs.append(random.choice(recommendations_dict["Physical_Activity_low"]))
        if user.get('Caffeine_Intake', 0) > 3:
            recs.append(random.choice(recommendations_dict["Caffeine_Intake_high"]))
        if user.get('Alcohol_Intake', 0) > 2:
            recs.append(random.choice(recommendations_dict["Alcohol_Intake_high"]))
        if user.get('Blood_Pressure', 0) > 130:
            recs.append(random.choice(recommendations_dict["Blood_Pressure_high"]))
        if user.get('Cholesterol_Level', 0) > 200:
            recs.append(random.choice(recommendations_dict["Cholesterol_high"]))
        if user.get('Blood_Sugar_Level', 0) > 110:
            recs.append(random.choice(recommendations_dict["Blood_Sugar_high"]))
    except Exception:
        pass
    if not recs:
        recs.append("You are maintaining a healthy balance! Keep it up 🎉")
    return recs