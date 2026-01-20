# Heart Disease Prediction App

An interactive web app to **predict the risk of heart disease** using **Machine Learning (K-Nearest Neighbors)**.  
Users can enter patient details to get a **real-time prediction and probability** of heart disease.  
Built with **Python**, **Streamlit**, **scikit-learn**, and **NumPy**.

---

## 🔹 Live Demo
Try the app online here: [Click to Open](https://hearthealthpred.streamlit.app/)

---

## 🔹 Dataset Overview
The app uses medical features commonly associated with heart disease.  
Each input feature is numeric or categorical as described below:

| Feature | Data Type | Description |
|---------|-----------|-------------|
| Age | Numeric | Patient age in years |
| Sex | Categorical (0 = Female, 1 = Male) | Gender of patient |
| Chest Pain Type | Categorical (0,1,2,3) | Type of chest pain |
| Resting Blood Pressure | Numeric | Systolic blood pressure (mm Hg) |
| Cholesterol | Numeric | Serum cholesterol (mg/dl) |
| Fasting Blood Sugar | Categorical (0 = <=120 mg/dl, 1 = >120 mg/dl) | Fasting blood sugar indicator |
| Resting ECG | Categorical (0,1,2) | Resting electrocardiographic results |
| Max Heart Rate | Numeric | Maximum heart rate achieved |
| Exercise Angina | Categorical (0 = No, 1 = Yes) | Angina induced by exercise |
| Oldpeak | Numeric | ST depression induced by exercise relative to rest |
| ST Slope | Categorical (0,1,2) | Slope of peak exercise ST segment |

**Target:**  
- 0 = No heart disease  
- 1 = Heart disease  

---
