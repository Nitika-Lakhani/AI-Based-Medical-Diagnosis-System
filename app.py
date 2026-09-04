import numpy as np
import pickle
from tensorflow import keras
from groq import Groq
import json
import os
from dotenv import load_dotenv

load_dotenv()


client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -------------------------------
# LOAD FILES
# -------------------------------
model = keras.models.load_model("model/disease_model.keras")

with open("model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("model/columns.pkl", "rb") as f:
    columns = pickle.load(f)

with open("model/disease_symptoms_map.pkl", "rb") as f:
    disease_symptoms_map = pickle.load(f)

# -------------------------------
# FUNCTION: Convert symptoms → vector
# -------------------------------
def preprocess_input(symptoms):
    input_vector = np.zeros(len(columns))

    for symptom in symptoms:
        symptom = symptom.lower().strip()
        if symptom in columns:
            index = columns.index(symptom)
            input_vector[index] = 1

    return input_vector.reshape(1, -1)

# -------------------------------
# FUNCTION: Predict disease
# -------------------------------
def predict_disease_final(user_symptoms):
    # Prepare input
    input_data = preprocess_input(user_symptoms)

    # Model prediction
    predictions = model.predict(input_data)[0]

    # Get top 30 candidates
    top_indices = np.argsort(predictions)[-30:][::-1]

    results = []

    # Dynamic threshold (better than 50%)
    threshold = max(2, int(0.4 * len(user_symptoms)))

    for i in top_indices:
        disease = le.inverse_transform([i])[0]
        confidence = float(predictions[i])

        # Get disease symptom vector
        disease_vector = disease_symptoms_map.loc[disease].values

        match_count = 0

        for symptom in user_symptoms:
            symptom = symptom.lower().strip()

            if symptom in columns:
                idx = columns.index(symptom)

                if disease_vector[idx] > 0:
                    match_count += 1

        # Apply filter
        if match_count >= threshold:
            # Optional weighted score (better ranking)
            score = (0.7 * confidence) + (0.3 * (match_count / len(user_symptoms)))

            results.append((disease, confidence, match_count, score))

    # If we have valid filtered results
    if len(results) > 0:
        results = sorted(results, key=lambda x: x[3], reverse=True)

        return [(results[0][0], results[0][1])]

    # Fallback (if nothing matched)
    i = top_indices[0]
    disease = le.inverse_transform([i])[0]
    confidence = float(predictions[i])

    return [(disease, confidence)]

    return fallback

# -------------------------------
# FUNCTION: Get AI Medical Advice
# -------------------------------
def get_medical_advice(diseases, symptoms):
    prompt = f"""
    A user has the following symptoms: {symptoms}
    The predicted diseases are: {diseases}

    Generate detailed medical guidance.

    STRICT REQUIREMENTS:
    - Return ONLY valid JSON
    - Include a short disease description (3–4 lines)
    - Each category MUST contain AT LEAST 5 points
    - Each point must be clear, and practical and little elaborated 
    - Do NOT leave any field empty
    - Return ONLY raw JSON (no ```json, no markdown)

    JSON FORMAT:

    {{
        "disease_info": "brief explanation of the disease",
        "precautions": ["point1", "point2", "point3", "point4", "point5"],
        "medications": ["point1", "point2", "point3", "point4", "point5"],
        "diet": ["point1", "point2", "point3", "point4", "point5"],
        "lifestyle": ["point1", "point2", "point3", "point4", "point5"]
    }}

    NOTES:
    - Keep disease_info simple and understandable
    - No technical jargon
    - Medications should be general (like paracetamol, antihistamines), NOT prescriptions 
    - Diet should include foods to eat and avoid
    - Lifestyle should include habits, sleep, exercise, hydration
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4   # slightly higher for richer output
    )

    import re

    text = response.choices[0].message.content.strip()
    # 🔥 Extract JSON safely
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        
        if json_match:
            clean_json = json_match.group(0)
            return json.loads(clean_json)
        else:
            return {"error": "No JSON found", "raw": text}

    except Exception as e:
        return {"error": str(e), "raw": text}

# -------------------------------
# FUNCTION: Basic advice 
# -------------------------------
def ensure_minimum(advice):
    default_data = {
        "precautions": [
            "Stay hydrated",
            "Get proper rest",
            "Avoid exposure to triggers",
            "Maintain hygiene",
            "Monitor symptoms regularly"
        ],
        "medications": [
            "Paracetamol for fever",
            "Antihistamines if allergy",
            "Cough suppressants if needed",
            "ORS for hydration",
            "Consult doctor if symptoms worsen"
        ],
        "diet": [
            "Eat light and nutritious food",
            "Include fruits and vegetables",
            "Avoid oily and spicy food",
            "Drink plenty of fluids",
            "Include vitamin C rich foods"
        ],
        "lifestyle": [
            "Get enough sleep",
            "Avoid stress",
            "Do light exercise",
            "Stay hydrated",
            "Maintain cleanliness"
        ]
    }

    for key in default_data:
        if key not in advice or len(advice[key]) < 5:
            advice[key] = default_data[key]

    return advice

# -------------------------------
# TEST RUN (ONLY FOR TERMINAL)
# -------------------------------
if __name__ == "__main__":

    test_cases = [
        ["fever", "cough", "chills"],
    ]

    for idx, symptoms in enumerate(test_cases, 1):
        print("\n" + "="*60)
        print(f"🧪 Test Case {idx}")
        print("Input Symptoms:", symptoms)
        print("-"*60)

        results = predict_disease_final(symptoms)

        if len(results) == 0:
            print("❌ No strong prediction found")
            continue

        print("🔍 Top Predictions:\n")

        for i, (disease, score) in enumerate(results, 1):
            print(f"{i}. {disease}")
            print(f"   Confidence: {score:.4f}")

        # 🔥 GET ADVICE
        disease_names = [results[0][0]]
        advice = get_medical_advice(disease_names, symptoms)

        if "error" in advice:
            print("⚠️ API issue, using fallback")

        advice = ensure_minimum(advice)

        print("\n🧠 AI Medical Advice:\n")

        if "error" in advice:
            print("⚠️ Error parsing advice:", advice["error"])
        else:
            print("Disease Info:", advice.get("disease_info", []))
            print("Precautions:", advice.get("precautions", []))
            print("Medications:", advice.get("medications", []))
            print("Diet:", advice.get("diet", []))
            print("Lifestyle:", advice.get("lifestyle", []))

        print("="*60)