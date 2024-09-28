# Importing Libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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