import streamlit as st
from app import predict_disease_final, get_medical_advice, ensure_minimum
import pickle

# -------------------------------
# LOAD SYMPTOMS
# -------------------------------
with open("model/columns.pkl", "rb") as f:
    symptom_columns = pickle.load(f)

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Medical Diagnosis AI", layout="wide")

# -------------------------------
# SIDEBAR DISCLAIMER
# -------------------------------
st.sidebar.info(
    "⚠️ This is an AI-powered tool for educational purposes only.\n\n"
    "Always consult a real doctor for medical advice."
)

# -------------------------------
# TITLE
# -------------------------------
st.title("🩺 Advanced Medical Diagnosis System")
st.write("Select your symptoms below for an AI-powered health assessment.")

# -------------------------------
# SYMPTOM SELECTOR
# -------------------------------
selected_symptoms = st.multiselect(
    "Search and select symptoms:",
    options=symptom_columns
)

# -------------------------------
# BUTTON
# -------------------------------
if st.button("Generate Diagnosis"):

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")
    else:
        # -------------------------------
        # PREDICTION
        # -------------------------------
        results = predict_disease_final(selected_symptoms)

        disease = results[0][0]
        confidence = results[0][1]

        # -------------------------------
        # AI ADVICE
        # -------------------------------
        advice = get_medical_advice([disease], selected_symptoms)
        advice = ensure_minimum(advice)

        # -------------------------------
        # LAYOUT (2 COLUMNS)
        # -------------------------------
        col1, col2 = st.columns([1, 2])

        # -------------------------------
        # LEFT PANEL (Diagnosis)
        # -------------------------------
        with col1:
            st.subheader("Diagnosis Results")
            st.write("## Predicted Disease")
            st.success(disease)

            st.write(f"Confidence Score: {confidence*100:.2f}%")
            st.progress(float(confidence))

        # -------------------------------
        # RIGHT PANEL (Details)
        # -------------------------------
        with col2:
            st.subheader("Description")

            st.info(advice.get("disease_info", "No description available"))

            # -------------------------------
            # TABS
            # -------------------------------
            tab1, tab2, tab3, tab4 = st.tabs(
                ["💊 Medications", "⚠️ Precautions", "🥗 Diet", "🏃 Lifestyle"]
            )

            with tab1:
                for item in advice["medications"]:
                    st.write(f"• {item}")

            with tab2:
                for item in advice["precautions"]:
                    st.write(f"• {item}")

            with tab3:
                for item in advice["diet"]:
                    st.write(f"• {item}")

            with tab4:
                for item in advice["lifestyle"]:
                    st.write(f"• {item}")