# Importing Libraries 
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

st.header("Diabetes Risk Calculator")

##Loading Data
df = pd.read_csv('diabetes_data.csv')
newdf = df

st.write("Our data is below:")
st.write(newdf.head(20))
#st.write("We will now filter this data to focus on South Asian subjects for our model.")
#train = train[train["Race"] == "Asian"]
#st.write(train)


def user_input_features():
  
    st.write("**Please fill out the questionnaire below to see if you are at risk of diabetes:**")
    st.write("""**1. Do you have high blood pressure?**""") 
    bp = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "a")
    if bp == "Yes":
        bp = 1.0
    elif bp == "No":
        bp == 0.0

    st.write("""**2. Do you have high cholestrol?**""") 
    cholestrol = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "b")
    if cholestrol == "Yes":
        cholestrol = 1.0
    elif cholestrol == "No":
        cholestrol == 0.0

    st.write("""**3. Have you had your cholestrol checked in the past two years?**""") 
    cholestrolCheck = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "c")
    if cholestrolCheck == "Yes":
        cholestrolCheck = 1.0
    elif cholestrolCheck == "No":
        cholestrolCheck == 0.0

    st.write("""**4. What is your BMI?**""") 
    bmi = st.slider('', 0.0, 50.0, 25.0, key = "d")
  
    st.write("""**5. Have you smoked over 100 cigarettes in your lifetime?**""") 
    smoke = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "e")
    if smoke == "Yes":
        smoke = 1.0
    elif smoke == "No":
        smoke == 0.0

    st.write("""**6. Have you ever had a stroke?**""") 
    stroke = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "f")
    if stroke == "Yes":
        stroke = 1.0
    elif stroke == "No":
        stroke == 0.0

    st.write("""**7. Have you ever been diagnosed with heart disease or had a heart attack?**""") 
    heartDisease = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "g")
    if heartDisease == "Yes":
        heartDisease = 1.0
    elif heartDisease == "No":
        heartDisease == 0.0

    st.write("""**8. Do you do at least 2.5 hours of moderate-intensity exercise per week?**""") 
    exercise = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "h")
    if exercise == "Yes":
        exercise = 1.0
    elif exercise == "No":
        exercise == 0.0

    st.write("""**9. Do you consume one serving of fruit per day?**""") 
    fruit = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "i")
    if fruit == "Yes":
        fruit = 1.0
    elif fruit == "No":
        fruit == 0.0

    st.write("""**10. Do you consume one serving of vegetables per day?**""") 
    vegetables = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "j")
    if vegetables == "Yes":
        vegetables = 1.0
    elif vegetables == "No":
        vegetables == 0.0

    st.write("""**11. Do you consume over 15 (men) or 8 (women) drinks per week?**""") 
    alc = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "k")
    if alc == "Yes":
        alc = 1.0
    elif alc == "No":
        alc == 0.0

    st.write("""**12. Do you have healthcare coverage?**""") 
    coverage = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "l")
    if coverage == "Yes":
        coverage = 1.0
    elif coverage == "No":
        coverage == 0.0

    st.write("""**13. Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?**""") 
    noDoc = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "m")
    if noDoc == "Yes":
        noDoc = 1.0
    elif noDoc == "No":
        noDoc == 0.0

    st.write("""**14. Rate your general health on a scale from 1-5**""") 
    genHealth = st.slider('', 1.0, 5.0, 2.0, key = "n")

    st.write("""**15. Of the last 30 days, how many would you consider \'bad\' days mentally?**""") 
    mental = st.slider('', 0.0, 30.0, 15.0, key = "o")

    st.write("""**16. Of the last 30 days, how many would you consider \'bad\' days physically?**""") 
    physical = st.slider('', 0.0, 30.0, 15.0, key = "p")

    st.write("""**17. Do you have serious difficultly walking or climbing stairs?**""") 
    walking = st.selectbox("Yes or No", ["Select Answer", "Yes", "No"], key = "q")
    if walking == "Yes":
        walking = 1.0
    elif walking == "No":
        walking == 0.0

    st.write("""**18. What is your gender?**""") 
    gender = st.selectbox("Male or Female", ["Select Answer", "Male", "Female"], key = "r")
    if gender == "Female":
        gender = 0.0
    elif gender == "Male":
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

    data = {"HighBP": bp, "HighChol": cholestrol, "CholCheck": cholestrolCheck, "BMI": bmi, "Smoker": smoke, "Stroke": stroke,
            "HeartDiseaseorAttack": heartDisease, "PhysActivity": exercise, "Fruits": fruit, "Veggies": vegetables, "HvyAlcoholConsump": alc,
            "AnyHealthcare": coverage, "NoDocbcCost": noDoc, "GenHlth": genHealth, "MentHlth": mental,
            "PhysHlth": physical, "DiffWalk": walking, "Sex": gender, "Age": age, "Education": edu,
            "Income": income}
    features = pd.DataFrame(data, index=[0])
    st.subheader('Given Inputs : ')
    st.write(features)

    return features

user = user_input_features()

##Transform data

discrete = ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
       'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies','HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
       'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']     

discrete2 = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
       'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies','HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
       'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']    

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
enc.fit(newdf[discrete])
newdf[discrete] = enc.transform(newdf[discrete])

enc.fit(user[discrete2])
user[discrete2] = enc.transform(user[discrete2])

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split

param = newdf.drop(columns=['Diabetes_binary']).values  # Adjust 'Diabetes_binary' based on the actual column name
# Target: the diabetes outcome column
target = newdf['Diabetes_binary'].values  # Use the correct column name for the target

X_train, X_test, y_train, y_test = train_test_split(param, target, test_size=0.1, random_state=12)

model = ExtraTreesClassifier()
model.fit(param, target)

prediction = model.predict(user)
st.subheader('Prediction using ExtraTreesClassifier:')
df1=pd.DataFrame(prediction,columns=['0'])
df1.loc[df1['0'] == 0, 'Chances of Diabetes'] = 'No'
df1.loc[df1['0'] == 1, 'Chances of Diabetes'] = 'Yes'
st.write(df1)

prediction_proba = model.predict_proba(user)
st.subheader('Prediction Probability in % :')
st.write(prediction_proba * 100)

model = RandomForestClassifier()
model.fit(param, target)

prediction = model.predict(user)
st.subheader('Prediction using RandomForestClassifer:')
df1=pd.DataFrame(prediction,columns=['0'])
df1.loc[df1['0'] == 0, 'Chances of Diabetes'] = 'No'
df1.loc[df1['0'] == 1, 'Chances of Diabetes'] = 'Yes'
st.write(df1)

prediction_proba = model.predict_proba(user)
st.subheader('Prediction Probability in % :')
st.write(prediction_proba * 100)