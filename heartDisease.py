import streamlit as st
import joblib
import numpy as np

model = joblib.load("heart_knn_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
st.title("Heart Disease Prediction")
st.write("Enter the following details to predict the presence of heart disease:")

def user_input_features():
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ("Male", "Female"))
    chest_pain_type = st.selectbox("Chest Pain Type", (0, 1, 2, 3))
    resting_bp_s = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", (0, 1))
    resting_ecg = st.selectbox("Resting ECG", (0, 1, 2))
    max_heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", (0, 1))
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
    st_slope = st.selectbox("ST Slope", (0, 1, 2))
    
    # Convert Sex to numeric
    sex = 1 if sex == "Male" else 0
    
    # Create numpy array for the model
    features = np.array([age, sex, chest_pain_type, resting_bp_s, cholesterol,
                        fasting_blood_sugar, resting_ecg, max_heart_rate,
                        exercise_angina, oldpeak, st_slope])
    return features.reshape(1, -1)
input_data = user_input_features()
input_data_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)
    
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.write("⚠️ The patient may have heart disease.")
    else:
        st.write("✅ The patient is likely healthy.")

    st.subheader("Prediction Probability:")
    st.write(f"Probability of no heart disease: {prediction_proba[0][0]*100:.2f}%")
    st.write(f"Probability of heart disease: {prediction_proba[0][1]*100:.2f}%")