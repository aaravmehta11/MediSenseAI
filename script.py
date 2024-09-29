import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st
import sklearn as sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,classification_report
from sklearn.metrics import f1_score


st.header("Arthritis Risk Calculator")

#Loading Data
df = pd.read_csv("arthritis.csv")
df = df.drop(df.columns[0], axis=1)
st.write("Our preliminary data is below: ")
st.write(df.head(30))

def user_input_features():
  
  st.write("**Please fill out the questionnaire below to see if you are at risk of diabetes:**")

  st.write("""**1. How do you feel about your general health?**""") 
  gen_health = st.selectbox("(Poor, Fair, Good, Very Good, Excellent)", ["Poor", "Fair", "Good", "Very Good", "Excellent"], key = 'a')

  st.write("""**2. When was the last time you had a medical checkup?**""") 
  checkup = st.selectbox("(Within a year, Within two years, Within five years, Over five years, never)", ["Within the past year", "Within the past 2 years", "Within the past 5 years", "5 or more years ago", "Never"], key = 'b')

  st.write("""**3. Do you exercise at all?**""") 
  exercise = st.selectbox("(Yes or No)", ["Yes", "No"], key = 'c')

  st.write("""**4. Have you been clincially diagnosed with depression?**""") 
  depression = st.selectbox("(Yes or No)", ["Yes", "No"], key = 'd')
  
  st.write("""**5. What is your sex?**""") 
  sex = st.selectbox("(Male or Female)", ["Male", "Female"], key = 'e')

  st.write("""**6. How old are you?**""") 
  age = st.slider('', 0, 100, 25, key = '1')
  
  age_cat= ""

  if age <= 24:
    age_cat = "18-24"
  elif age <= 29:
    age_cat = "25-29"
  elif age <= 34:
    age_cat = "30-34"
  elif age <= 39:
    age_cat = "35-39"
  elif age <= 44:
    age_cat = "40-44"
  elif age <= 49:
    age_cat = "45-49"
  elif age <= 54:
    age_cat = "50-54" 
  elif age <= 59:
    age_cat = "55-59"
  elif age <= 64:
    age_cat = "60-64"  
  elif age <= 69:
    age_cat = "65-69"  
  elif age <= 74:
    age_cat = "70-74" 
  elif age <= 79:
    age_cat = "75-79"  
  else:
    age_cat = "80+"

  st.write("""**7. What is your height (cm)?**""") 
  height = st.slider('', 50.0, 250.0, 150.0, key = '2') 

  st.write("""**8. What is your weight (kg)?**""") 
  weight = st.slider('', 10.0, 200.0, 70.0, key = '3')
  
  st.write("""**9. What is your BMI?**""") 
  BMI = st.slider('', 10.0, 100.0, 30.0, key = '4')

  st.write("""**10. Do you smoke?**""") 
  smoke = st.selectbox("(Yes or No)", ["Yes", "No"], key = 'f')
  
  st.write("""**11. On a scale from 0-30, how much alcohol do you consume?**""") 
  alc = st.slider('', 0, 30, 0, key = '5')
  
  st.write("""**12. On a scale from 0-60, what is your level of fruit consumption?**""") 
  fruit = st.slider('', 0, 60, 30, key = '6')  

  st.write("""**13. On a scale from 0-30, what is your level of vegetable consumption?**""") 
  veg = st.slider('', 0, 30, 15, key = '7')
  
  st.write("""**14. On a scale from 0-30, how much fried food do you consume?**""") 
  fried = st.slider('', 0, 30, 15, key = '8')
  
  st.write("""**15. Do you have any form of heart disease?**""") 
  heart = st.selectbox("(Yes or No)", ["Yes", "No"], key = 'g')
  
  st.write("""**16. Do you have skin cancer?**""") 
  skin = st.selectbox("(Yes or No)", ["Yes", "No"], key = 'i')
  
  st.write("""**17. Do you any other form of cancer?**""") 
  cancer = st.selectbox("(Yes or No)", ["Yes", "No"], key = 'j')
  
  st.write("""**18. Do you have diabetes**""") 
  diabetes = st.selectbox("(Yes or No)", ["Yes", "No"], key = 'k') 

  data = {'General_Health': gen_health, 'Checkup': checkup, 'Exercise': exercise, "Heart_Disease": heart, "Skin_Cancer": skin, "Other_Cancer": cancer, 'Depression': depression, "Diabetes": diabetes, 'Sex': sex, 'Age_Category': age_cat, 'Height_(cm)': height, 'Weight_(kg)': weight, 'BMI': BMI, 'Smoking_History': smoke, 'Alcohol_Consumption': alc, 'Fruit_Consumption': fruit, 'Green_Vegetables_Consumption': veg,'FriedPotato_Consumption': fried}
  features = pd.DataFrame(data, index=[0])
  st.subheader('Given Inputs : ')
  st.write(features)
  
  return features

user = user_input_features()

newdf = df
discrete = ["General_Health", "Checkup", "Exercise", "Heart_Disease", "Skin_Cancer", "Other_Cancer", "Depression", "Diabetes", "Arthritis", "Sex", "Age_Category", "Smoking_History"]
discrete2 = ["General_Health", "Checkup", "Exercise", "Heart_Disease", "Skin_Cancer", "Other_Cancer", "Depression", "Diabetes", "Sex", "Age_Category", "Smoking_History"]


#Transform Data
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
enc.fit(newdf[discrete])
newdf[discrete] = enc.transform(newdf[discrete])

enc.fit(user[discrete2])
user[discrete2] = enc.transform(user[discrete2])

X = df.drop('Arthritis', axis=1)  # Replace 'arthritis' with your target variable name
y = df['Arthritis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Accuracy:")
st.write(f'Accuracy: {accuracy * 100:.2f}%')

new_person_scaled = scaler.transform(user)
predicted_probabilities = model.predict_proba(new_person_scaled)
diabetes_probability = predicted_probabilities[0][1]  # Probability of class 1 (diabetes)
percentage_chance = diabetes_probability * 100
st.subheader('Prediction Probability in % :')
st.write(f'Predicted risk of diabetes: {percentage_chance:.2f}%')