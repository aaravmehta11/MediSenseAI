import pandas as pd
import numpy as np
import streamlit as st
import sklearn as sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import imblearn as imblearn
from imblearn.over_sampling import SMOTE


#Streamlit Page layout
st.set_page_config(
    page_title= "Arthritis Risk Calculator",
    layout="centered")
st.header("Arthritis Risk Calculator")

#Loading Data
df = pd.read_csv("arthritis.csv")
df = df.drop(df.columns[0], axis=1)
st.write("Our preliminary data is below: ")
st.write(df.head(30))

#Defining a method to collect user data throught streamlit interface
def user_input_features():
  
  st.write("**Please fill out the questionnaire below to see if you are at risk of diabetes:**")

  st.write("""**1. How do you feel about your general health?**""") 
  gen_health = st.selectbox("(Poor, Fair, Good, Very Good, Excellent)", ["Poor", "Fair", "Good", "Very Good", "Excellent"], key = 'a')

  st.write("""**2. When was the last time you had a medical checkup?**""") 
  checkup = st.selectbox("(Never, Over five years, Within five years, Within two years, Within a year)", ["Never", "Over five years", "Within five years", "Within two years", " Within a year"], key = 'b')

  st.write("""**3. Do you exercise at all?**""") 
  exercise = st.selectbox("(Yes or No)", ["Yes", "No"], key = 'c')

  st.write("""**4. Have you been clinically diagnosed with depression?**""") 
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
  
  st.write("""**18. Do you have diabetes?**""") 
  diabetes = st.selectbox("(Yes or No)", ["Yes", "No"], key = 'k') 

  data = {'General_Health': gen_health, 'Checkup': checkup, 'Exercise': exercise, "Heart_Disease": heart, "Skin_Cancer": skin, "Other_Cancer": cancer, 'Depression': depression, "Diabetes": diabetes, 'Sex': sex, 'Age_Category': age_cat, 'Height_(cm)': height, 'Weight_(kg)': weight, 'BMI': BMI, 'Smoking_History': smoke, 'Alcohol_Consumption': alc, 'Fruit_Consumption': fruit, 'Green_Vegetables_Consumption': veg,'FriedPotato_Consumption': fried}
  features = pd.DataFrame(data, index=[0])
  st.subheader('Given Inputs : ')
  st.write(features)
  
  return features

user = user_input_features()

#Every discrete variable 
discrete = ["General_Health", "Checkup", "Exercise", "Heart_Disease", "Skin_Cancer", "Other_Cancer", "Depression", "Diabetes", "Arthritis", "Sex", "Age_Category", "Smoking_History"]

#Transform Data into all numbers
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()

#Transforming original dataset
for col in df.columns:
  df[col] = enc.fit_transform(df[[col]])

#Transforming user input
for col in user.columns:
  if col in discrete:
    user[col] = enc.fit_transform(user[[col]])

#Using SMOTE to account for oversampling in our model
X = df.drop('Arthritis', axis=1)  # Replace 'arthritis' with your target variable name
y = df['Arthritis']
sm = SMOTE(random_state = 500)#sm.fit(X,y)
x_resem, y_resem = sm.fit_resample(X, y)

#Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(x_resem, y_resem, test_size=0.15, random_state=200)

#Using a RandomForestClassifier for our model
model = RandomForestClassifier(random_state = 42, n_estimators = 150, class_weight = "balanced")
model.fit(X_train, y_train)
model.score(X_test, y_test)
  
#Accuracy measure of our RandomForestClassifier
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.subheader(f"Model Accuracy: {accuracy * 100:.2f}%")

#Output probability of Arthritis 
predicted_probabilities = model.predict_proba(user)
arthritis_probability = predicted_probabilities[0][1]  # Probability of class 1 (diabetes)
percentage_chance = arthritis_probability * 100
st.subheader(f'Probability of Arthritis: {percentage_chance:.2f}%')
