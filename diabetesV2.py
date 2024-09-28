# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

bp = 0
cholestrol = 0
cholestrolCheck = 0
physical = 0
smoke = 0
stroke = 0
heartDisease = 0
exercise = 0
fruit = 0
vegetables = 0
alc = 0
coverage = 0
noDoc = 0
genHealth = 0
mental = 0
physical = 0
walking = 0
gender = 0
age = 0
edu = 0
income = 0


st.header("Diabetes Risk Calculator")

st.write("**Please fill out the questionnaire below to see if you are at risk of diabetes:**")
st.write("""**1. Do you have high blood pressure?**""") 
bpSelector = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "a")
if bpSelector == "Yes":
    bp = 1.0
elif bpSelector == "No":
    bp == 0.0

st.write("""**2. Do you have high cholestrol?**""") 
c = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "b")
if c == "Yes":
    cholestrol = 1.0
elif c == "No":
    cholestrol == 0.0

st.write("""**3. Have you had your cholestrol checked in the past two years?**""") 
cc = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "c")
if cc == "Yes":
    cholestrolCheck = 1.0
elif cc == "No":
    cholestrolCheck == 0.0

st.write("""**4. What is your BMI?**""") 
bmi = st.slider('', 0.0, 50.0, 25.0, key = "d")

st.write("""**5. Have you smoked over 100 cigarettes in your lifetime?**""") 
sm = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "e")
if sm == "Yes":
    smoke = 1.0
elif sm == "No":
    smoke == 0.0

st.write("""**6. Have you ever had a stroke?**""") 
str = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "f")
if str == "Yes":
    stroke = 1.0
elif str == "No":
    stroke == 0.0

st.write("""**7. Have you ever been diagnosed with heart disease or had a heart attack?**""") 
hd = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "g")
if hd == "Yes":
    heartDisease = 1.0
elif hd == "No":
    heartDisease == 0.0

st.write("""**8. Do you do at least 2.5 hours of moderate-intensity exercise per week?**""") 
e = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "h")
if e == "Yes":
    exercise = 1.0
elif e == "No":
    exercise == 0.0

st.write("""**9. Do you consume one serving of fruit per day?**""") 
f = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "i")
if f == "Yes":
    fruit = 1.0
elif f == "No":
    fruit == 0.0

st.write("""**10. Do you consume one serving of vegetables per day?**""") 
v = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "j")
if v == "Yes":
    vegetables = 1.0
elif v == "No":
    vegetables == 0.0

st.write("""**11. Do you consume over 15 (men) or 8 (women) drinks per week?**""") 
alcohol = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "k")
if alcohol == "Yes":
    alc = 1.0
elif alcohol == "No":
    alc == 0.0

st.write("""**12. Do you have healthcare coverage?**""") 
cov = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "l")
if cov == "Yes":
    coverage = 1.0
elif cov == "No":
    coverage == 0.0

st.write("""**13. Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?**""") 
nd = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "m")
if nd == "Yes":
    noDoc = 1.0
elif nd == "No":
    noDoc == 0.0

st.write("""**14. Rate your general health on a scale from 1-5**""") 
genHealth = st.slider('', 1.0, 5.0, 2.0, key = "n")

st.write("""**15. Of the last 30 days, how many would you consider \'bad\' days mentally?**""") 
mental = st.slider('', 0.0, 30.0, 15.0, key = "o")

st.write("""**16. Of the last 30 days, how many would you consider \'bad\' days physically?**""") 
physical = st.slider('', 0.0, 30.0, 15.0, key = "p")

st.write("""**17. Do you have serious difficultly walking or climbing stairs?**""") 
w = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "q")
if w == "Yes":
    walking = 1.0
elif w == "No":
    walking == 0.0

st.write("""**18. What is your gender?**""") 
gen = st.selectbox("Male or Female", ["Select Answer", "Male", "Female"], key = "r")
if gen == "Female":
    gender = 0.0
elif gen == "Male":
    gender == 1.0

st.write("""**19. What is your age?**""") 
age = st.slider('', 0, 100, 50, key = "s")

if 18 <= age <= 24:
    age = 1.0  # 18-24
elif 25 <= age <= 29:
    age = 2.0  # 25-29
elif 30 <= age <= 34:
    age = 3.0  # 30-34
elif 35 <= age <= 39:
    age = 4.0  # 35-39
elif 40 <= age <= 44:
    age = 5.0  # 40-44
elif 45 <= age <= 49:
    age = 6.0  # 45-49
elif 50 <= age <= 54:
    age = 7.0  # 50-54
elif 55 <= age <= 59:
    age = 8.0  # 55-59
elif 60 <= age <= 64:
    age = 9.0  # 60-64
elif 65 <= age <= 69:
    age = 10.0  # 65-69
elif 70 <= age <= 74:
    age = 11.0  # 70-74
elif 75 <= age <= 79:
    age = 12.0  # 75-79
elif age >= 80:
    age = 13.0  # 80 or older

st.write("""**20. What is your education level on a scale 1-6?**""") 
edu = st.slider('', 0.0, 6.0, 3.0, key = "t")

st.write("""**21. What is your annual income (in thousands)?**""") 
income = st.slider('', 0.0, 100.0, 50.0, key = "u")

if income < 10:
    income = 1.0  # Less than 10k
elif 10 <= income < 15:
    income = 2.0  # 10k to 14.999k
elif 15 <= income < 20:
    income = 3.0  # 15k to 19.999k
elif 20 <= income < 25:
    income = 4.0  # 20k to 24.999k
elif 25 <= income < 35:
    income = 5.0  # 25k to 34.999k
elif 35 <= income < 50:
    income = 6.0  # 35k to 49.999k
elif 50 <= income < 75:
    income = 7.0  # 50k to 74.999k
elif income >= 75:
    income = 8.0  # 75k or more


# 1. Load the dataset
data = pd.read_csv('diabetes_data.csv')

# 2. Preprocess the data
X = data.drop(columns='Diabetes_binary')  # Drop the target column
y = data['Diabetes_binary']  # Target column (1 for diabetes, 0 for no diabetes)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy:")
st.write(f'Accuracy: {accuracy * 100:.2f}%')

# 5. Make predictions for a new person (replace with real input values)
new_person = [[bp, cholestrol, cholestrolCheck, physical, smoke, stroke, heartDisease, exercise, fruit, vegetables, alc, coverage,
               noDoc, genHealth, mental, physical, walking, gender, age, edu, income]]  # Example feature values
new_person_scaled = scaler.transform(new_person)

# Get predicted probabilities
predicted_probabilities = model.predict_proba(new_person_scaled)

# The probability of the positive class (having diabetes)
diabetes_probability = predicted_probabilities[0][1]  # Probability of class 1 (diabetes)
percentage_chance = diabetes_probability * 100

st.subheader('Prediction Probability in % :')
st.write(f'Predicted risk of diabetes: {percentage_chance:.2f}%')