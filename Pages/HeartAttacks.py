import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
import sklearn as sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

st.header("Heart Attack Risk Calculator")

df = pd.read_csv("heart_attack_prediction_dataset.csv")
newdf = df
newdf = newdf.drop(newdf.columns[0], axis = 1)
newdf = newdf.drop(newdf.columns[21:23], axis = 1)

st.write("**A data sample is shown below:**")
st.write(newdf.head(21))

def user_input_features():
    st.write("**Please fill out the questionare below to see if you are at risk of a heart attack:**")
    
    age = st.text_input("**1. Enter age:**", value = "0", key = 1) #continuous
    
    gender = st.selectbox("**2. Male or Female**", ["Male", "Female"], key = 2)
    
    st.write("**3. Enter cholesterol level:**")
    chl = st.slider('', 0, 500, 250, key = "x") #continuous
    
    st.write("**4. Enter systolic blood pressure:**") #continuous
    bp = st.slider('', 0, 240, 120, key=8)
    
    hr = st.text_input("**5. Enter heart rate:**", value = "0", key = "d") #continuous
    
    db = st.selectbox("**6. Diabetic?**", ["Yes", "No"], key = "e")
    
    fh = st.selectbox("**7. Family history of heart-related diseases?**", ["Yes", "No"], key = "f")
    
    smoker = st.selectbox("**8. Smoker Status:**", ["Yes", "No"], key = "g")
    
    obese = st.selectbox("**9. Obesity Status:**", ["Yes", "No"], key = "h")
    
    alc = st.selectbox("**10. Level of Alcohol Consumption:**", ["None", "Light", "Moderate", "Heavy"], key = "i")
    
    st.write("**11. Exercise Hours Per Week:**")
    exercise = st.slider('', 0, 20, 10, key = "j")
    
    diet = st.selectbox("**12. Diet Type**", ["Healthy", "Average", "Unhealthy"], key = "k")
    
    php = st.selectbox("**13. Previous Heart Problems?**", ["Yes", "No"], key = "l")
    
    meduse = st.selectbox("**14. Do you use any medication?**", ["Yes", "No"], key = "m")
    
    st.write("**15. Approximate stress level:**")
    stress = st.slider('', 1, 10, 5, key = "n")
    
    st.write("**16. Sedentary Hours in the Day:**")
    sedhours = st.slider('', 1, 24, 12, key = "o")
    
    st.write("**17. Annual Income:**")
    income = st.slider('', 0, 300000, 150000, key = "p")
    
    st.write("**18. Enter BMI:**")
    BMI = st.slider('', 1, 40, 20, key = "q")
    
    st.write("**19. Triglyceride Level:**")
    triglycerides = st.slider('', 0, 800, 400, key = "r")
    
    st.write("**20. How many days of the week do partake in physical activity?:**")
    activityperweek = st.slider('', 0, 7, 3, key = "s")
    
    st.write("**21. Hours of Sleep per Night:**")
    sleephrs = st.slider('', 0, 24, 12, key = "t")
    
    hem = st.selectbox("**22. Enter Hemisphere of Residence:**", ["Eastern", "Western", "Northern", "Southern"], key = "u")
    
    data = {"Age": age, "Sex": gender, "Cholesterol": chl, "Blood Pressure": bp, "Heart Rate": hr, "Diabetes": db, "Family History": fh, "Smoking": smoker, "Obesity": obese, "Alcohol Consumption": alc, "Exercise Hours Per Week": exercise, "Diet": diet, "Previous Heart Problems": php, "Medication Use": meduse, "Stress Level": stress, "Sedentary Hours Per Day": sedhours, "Income": income, "BMI": BMI, "Triglycerides": triglycerides, "Physical Activity Days Per Week": activityperweek, "Sleep Hours Per Day": sleephrs, "Hemisphere": hem}
    dframe = pd.DataFrame(data, index=[0])
    return dframe 
user = user_input_features()


# 1. Load the dataset (already done, newdf)

discrete = ["Sex", "Diabetes", "Family History", "Smoking", "Obesity", "Alcohol Consumption", "Diet", "Previous Heart Problems", "Medication Use", "Hemisphere"]
# 2. Preprocess the data

enc = OrdinalEncoder()
for col in newdf.columns:
    newdf[col] = enc.fit_transform(newdf[[col]])
    
for col in user.columns:
    if col in discrete:
        user[col] = enc.fit_transform(user[[col]])

X = newdf.drop("Heart Attack Risk", axis=1) # Drop the target column 'Heart Attack Risk'
y = newdf['Heart Attack Risk'] # Target column (1 for yes, 0 for no)

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#smote the dataset to balance out any overrepresentations
# Normalize the data (makes the mean = 0 and SD = 1, making each row comparable. does not change skewness. )
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)

# 3. Train the Logistic Regression Model

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
model.score(X_test, y_test)

# 4. Evaluate the Model
y_pred = model.predict(X_test) #use X_test because this portion of the data is used to evaluate how well the model generalizes to unseen data.
accuracy = accuracy_score(y_test, y_pred) #calculates the proprtion of correct predictions (y_pred to y_test) made by the model

st.subheader(f'Model Accuracy: {accuracy * 100:.2f}%')

#5. Make predictions for a new person (replace with real input values)
new_person_scaled = scaler.transform(user.values)  #transforms 'user' the same way the training data was scaled (adjusting x-bar and SD)

# Note:
  # Training data is split (80/20, 70/30 etc.). Most of the data is used to train the model by recognizing patterns, and the leftover data is used as a comparison to new, unseen, computer-generated inputs and their respective outputs. 
    #overfitted = model performs well on training data but poorly on testing data
    #underfittded = failing to capture underlying patterns in both datasets

#creates a 1x2 matrix: col1: prob(no), col2: prob(yes)
predicted_probabilities = model.predict_proba(new_person_scaled) 


#transform() --> transforms the data the same way the training data was scaled (adjusting mean and SD)
#fit_transform() --> fits (learns SD and mean) and makes the appropriate transformation. Used on training data initially.

heartattack_probability = predicted_probabilities[0][1] 
percentage_chance = heartattack_probability * 100



st.subheader(f'Predicted risk of a heart attack is {percentage_chance:.2f}%')






    



  





















    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

