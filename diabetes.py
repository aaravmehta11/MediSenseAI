# Importing Libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

st.header("Diabetes Risk Calculator")

##Loading Data
df = pd.read_csv('data/heart_2020_cleaned.csv')
newdf = df

st.write("Our data is below:")
st.write(newdf.head(20))
#st.write("We will now filter this data to focus on South Asian subjects for our model.")
#train = train[train["Race"] == "Asian"]
#st.write(train)

def user_input_features():
  
    st.write("**Please fill out the questionnaire below to see if you are at risk of diabetes:**")
    st.write("""**1. Do you have high blood pressure?**""") 
    BP = st.selectbox("(Yes or No", ["Yes", "No"], key = "a")
    st.write("""**You selected this option:**""",BP)
    if BP == "Yes":
        BP = 1
    elif BP == "No":
        BP == 0

    st.write("""**2. Do you have high cholestrol?**""") 
    cholestrol = st.selectbox("(Yes or No", ["Yes", "No"], key = "a")
    st.write("""**You selected this option:**""",cholestrol)
    if cholestrol == "Yes":
        cholestrol = 1.0
    elif cholestrol == "No":
        cholestrol == 0.0

    st.write("""**3. Have you had your cholestrol checked in the past two years?**""") 
    cholestrolCheck = st.selectbox("(Yes or No", ["Yes", "No"], key = "a")
    st.write("""**You selected this option:**""",cholestrolCheck)
    if cholestrolCheck == "Yes":
        cholestrolCheck = 1.0
    elif cholestrolCheck == "No":
        cholestrolCheck == 0.0

    st.write("""**4. What is your BMI?**""") 
    physical = st.slider('', 0, 50, 25)
    st.write("""**You selected this option:**""", physical)
  
    st.write("""**5. Have you smoked over 100 cigarettes in your lifetime?**""") 
    smoke = st.selectbox("(Yes or No", ["Yes", "No"], key = "a")
    st.write("""**You selected this option:**""",smoke)
    if smoke == "Yes":
        smoke = 1.0
    elif smoke == "No":
        smoke == 0.0

    st.write("""**6. Have you ever had a stroke?**""") 
    stroke = st.selectbox("(Yes or No", ["Yes", "No"], key = "a")
    st.write("""**You selected this option:**""",stroke)
    if stroke == "Yes":
        stroke = 1.0
    elif stroke == "No":
        stroke == 0.0

    st.write("""**7. Have you ever been diagnosed with heart disease or had a heart attack?**""") 
    heartDisease = st.selectbox("(Yes or No", ["Yes", "No"], key = "a")
    st.write("""**You selected this option:**""",heartDisease)
    if heartDisease == "Yes":
        heartDisease = 1.0
    elif heartDisease == "No":
        heartDisease == 0.0

    st.write("""**8. Do you do at least 2.5 hours of moderate-intensity exercise per week?**""") 
    exercise = st.selectbox("(Yes or No", ["Yes", "No"], key = "a")
    st.write("""**You selected this option:**""",exercise)
    if exercise == "Yes":
        exercise = 1.0
    elif exercise == "No":
        exercise == 0.0

    st.write("""**9. Do you consume one serving of fruit per day?**""") 
    fruit = st.selectbox("(Yes or No", ["Yes", "No"], key = "a")
    st.write("""**You selected this option:**""",fruit)
    if fruit == "Yes":
        fruit = 1.0
    elif fruit == "No":
        fruit == 0.0

    st.write("""**10. Do you consume one serving of vegetables per day?**""") 
    vegetables = st.selectbox("(Yes or No", ["Yes", "No"], key = "a")
    st.write("""**You selected this option:**""",vegetables)
    if vegetables == "Yes":
        vegetables = 1.0
    elif vegetables == "No":
        vegetables == 0.0

    st.write("""**11. Do you consume over 15 (men) or 8 (women) drinks per week?**""") 
    alc = st.selectbox("(Yes or No", ["Yes", "No"], key = "a")
    st.write("""**You selected this option:**""",alc)
    if alc == "Yes":
        alc = 1.0
    elif alc == "No":
        alc == 0.0

    st.write("""**12. Do you have healthcare coverage?**""") 
    coverage = st.selectbox("(Yes or No", ["Yes", "No"], key = "a")
    st.write("""**You selected this option:**""",coverage)
    if coverage == "Yes":
        coverage = 1.0
    elif coverage == "No":
        coverage == 0.0

    st.write("""**13. Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?**""") 
    noDoc = st.selectbox("(Yes or No", ["Yes", "No"], key = "a")
    st.write("""**You selected this option:**""",noDoc)
    if noDoc == "Yes":
        noDoc = 1.0
    elif noDoc == "No":
        noDoc == 0.0

    st.write("""**14. Rate your general health on a scale from 1-5**""") 
    genHealth = st.slider('', 1, 5, 2.5)
    st.write("""**You selected this option:**""", genHealth)

    st.write("""**15. Of the last 30 days, how many would you consider \'bad\' days mentally?**""") 
    mental = st.slider('', 0, 30, 15)
    st.write("""**You selected this option:**""", mental)

    st.write("""**16. Of the last 30 days, how many would you consider \'bad\' days physically?**""") 
    physical = st.slider('', 0, 30, 15)
    st.write("""**You selected this option:**""", physical)

    st.write("""**17. Do you have serious difficultly walking or climbing stairs?**""") 
    walking = st.selectbox("(Yes or No", ["Yes", "No"], key = "a")
    st.write("""**You selected this option:**""",walking)
    if walking == "Yes":
        walking = 1.0
    elif walking == "No":
        walking == 0.0

    st.write("""**18. What is your gender?**""") 
    gender = st.selectbox("(Male or Female", ["Male", "Female"], key = "a")
    st.write("""**You selected this option:**""",gender)
    if gender == "Female":
        gender = 0.0
    elif gender == "Male":
        gender == 1.0

    st.write("""**19. What is your age?**""") 
    age = st.slider('', 0, 100, 50)
    st.write("""**You selected this option:**""", age)

    if 18 <= age <= 24:
        age = 1  # 18-24
    elif 25 <= age <= 29:
        age = 2  # 25-29
    elif 30 <= age <= 34:
        age = 3  # 30-34
    elif 35 <= age <= 39:
        age = 14  # 35-39
    elif 40 <= age <= 44:
        age = 5  # 40-44
    elif 45 <= age <= 49:
        age = 6  # 45-49
    elif 50 <= age <= 54:
        age = 7  # 50-54
    elif 55 <= age <= 59:
        age = 8  # 55-59
    elif 60 <= age <= 64:
        age = 9  # 60-64
    elif 65 <= age <= 69:
        age = 10  # 65-69
    elif 70 <= age <= 74:
        age = 11  # 70-74
    elif 75 <= age <= 79:
        age = 12  # 75-79
    elif age >= 80:
        age = 13  # 80 or older

    data = {'BMI': BMI, 'Smoking': smoke, 'AlcoholDrinking': alc, 'Stroke': stroke, 'PhysicalHealth': physical, 'MentalHealth': mental, 'DiffWalking': climb, 'Sex': sex, 'AgeCategory': age_cat, 'Race': 'Asian', 'Diabetic': diabetes, 'PhysicalActivity': exercise, 'GenHealth': gen_health,'SleepTime': sleep, 'Asthma': asthma, 'KidneyDisease': kidney, 'SkinCancer': cancer}
    features = pd.DataFrame(data, index=[0])
    st.subheader('Given Inputs : ')
    st.write(features)


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
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))

# 5. Make predictions for a new person (replace with real input values)
new_person = [[1.0, 1.0, 1.0, 30.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 5.0, 30.0, 30.0, 1.0, 0.0, 9.0, 5.0, 0.0]]  # Example feature values
new_person_scaled = scaler.transform(new_person)
prediction = model.predict(new_person_scaled)
print(f'Predicted risk of diabetes: {"Yes" if prediction[0] == 1 else "No"}')