# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # Import SMOTE
import streamlit as st

# Function to collect user input
def user_input_features():
    st.header("Diabetes Risk Calculator")
    st.write("**Please fill out the questionnaire below to see if you are at risk of diabetes:**")
    
    bp = st.selectbox("Do you have high blood pressure?", ["Yes", "No"])
    bp = 1.0 if bp == "Yes" else 0.0

    cholestrol = st.selectbox("Do you have high cholesterol?", ["Yes", "No"])
    cholestrol = 1.0 if cholestrol == "Yes" else 0.0

    cholestrolCheck = st.selectbox("Have you had your cholesterol checked in the past two years?", ["Yes", "No"])
    cholestrolCheck = 1.0 if cholestrolCheck == "No" else 0.0

    bmi = st.slider("What is your BMI?", 0.0, 50.0, 25.0)

    smoke = st.selectbox("Have you smoked over 100 cigarettes in your lifetime?", ["Yes", "No"])
    smoke = 1.0 if smoke == "Yes" else 0.0

    stroke = st.selectbox("Have you ever had a stroke?", ["Yes", "No"])
    stroke = 1.0 if stroke == "No" else 0.0

    heartDisease = st.selectbox("Have you ever been diagnosed with heart disease or had a heart attack?", ["Yes", "No"])
    heartDisease = 1.0 if heartDisease == "Yes" else 0.0

    exercise = st.selectbox("Do you do at least 2.5 hours of moderate-intensity exercise per week?", ["Yes", "No"])
    exercise = 1.0 if exercise == "No" else 0.0

    fruit = st.selectbox("Do you consume one serving of fruit per day?", ["Yes", "No"])
    fruit = 1.0 if fruit == "Yes" else 0.0

    vegetables = st.selectbox("Do you consume one serving of vegetables per day?", ["Yes", "No"])
    vegetables = 1.0 if vegetables == "No" else 0.0

    alc = st.selectbox("Do you consume over 15 (men) or 8 (women) drinks per week?", ["Yes", "No"])
    alc = 1.0 if alc == "No" else 0.0

    coverage = st.selectbox("Do you have healthcare coverage?", ["Yes", "No"])
    coverage = 1.0 if coverage == "No" else 0.0

    noDoc = st.selectbox("Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?", ["Yes", "No"])
    noDoc = 1.0 if noDoc == "No" else 0.0

    genHealth = st.slider("Rate your general health on a scale from 1 (Excellent) to 5 (Very Poor)", 1.0, 5.0, 2.0)

    mental = st.slider("Of the last 30 days, how many would you consider 'good' days mentally?", 0.0, 30.0, 15.0)

    physical = st.slider("Of the last 30 days, how many would you consider 'good' days physically?", 0.0, 30.0, 15.0)

    walking = st.selectbox("Do you have serious difficulty walking or climbing stairs?", ["Yes", "No"])
    walking = 1.0 if walking == "Yes" else 0.0

    gender = st.selectbox("What is your gender?", ["Female", "Male"])
    gender = 1.0 if gender == "Male" else 0.0

    age = st.slider("What is your age?", 0.0, 100.0, 50.0)

    edu = st.slider("What is your education level on a scale of 1-6?", 1.0, 6.0, 3.0)

    income = st.slider("What is your annual income (in thousands)?", 0.0, 100.0, 50.0)
    
    # Combine all inputs into a single DataFrame
    data = {
        "HighBP": bp, "HighChol": cholestrol, "CholCheck": cholestrolCheck, "BMI": bmi, "Smoker": smoke,
        "Stroke": stroke, "HeartDiseaseorAttack": heartDisease, "PhysActivity": exercise, "Fruits": fruit, 
        "Veggies": vegetables, "HvyAlcoholConsump": alc, "AnyHealthcare": coverage, "NoDocbcCost": noDoc, 
        "GenHlth": genHealth, "MentHlth": mental, "PhysHlth": physical, "DiffWalk": walking, 
        "Sex": gender, "Age": age, "Education": edu, "Income": income
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Load dataset and train Logistic Regression model with SMOTE
@st.cache(allow_output_mutation=True)
def load_and_train_model():
    # Load dataset
    data = pd.read_csv('data/diabetes_data.csv')
    target = 'Diabetes_binary'
    X = data.drop(columns=[target])
    y = data[target]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE to handle class imbalance only on the training set
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Scale the data
    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test = scaler.transform(X_test)

    # Train a Logistic Regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_resampled, y_train_resampled)

    return model, scaler, X_test, y_test

with st.spinner('Please wait...'):
    model, scaler, X_test, y_test = load_and_train_model()

# Collect user input and make predictions
user = user_input_features()
user_scaled = scaler.transform(user)

# Make predictions for the user input
predicted_probabilities = model.predict_proba(user_scaled)
diabetes_probability = predicted_probabilities[0][1]  # Probability of class 1 (diabetes)
percentage_chance = diabetes_probability * 100

# Display the predicted probability
st.subheader('Prediction Probability in % :')
st.write(f'Predicted risk of diabetes: {percentage_chance:.2f}%')

# Determine the risk category and show appropriate message
if percentage_chance < 30:
    risk_level = "Low Risk"
    bar_color = 'green'
elif 30 <= percentage_chance < 60:
    risk_level = "Moderate Risk"
    bar_color = 'orange'
else:
    risk_level = "High Risk"
    bar_color = 'red'

# Display the progress bar
st.write(f'Risk Level: {risk_level}')
progress_bar = st.progress(0)
progress_bar.progress(int(percentage_chance)) 