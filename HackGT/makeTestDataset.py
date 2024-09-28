import pandas as pd

# Read the CSV file
df = pd.read_csv("symbipredict_2022.csv")

symptom_columns = df.columns[:-1]  # Assuming 'prognosis' is the last column

# Function to convert each row of symptoms into a text description
def create_symptom_description(row):
    symptoms = [col.replace('_', ' ') for col in symptom_columns if row[col] == 1]
    if symptoms:
        return "The patient has " + ", ".join(symptoms) + "."
    else:
        return "The patient has no major symptoms."

# Apply the function to each row in your dataframe
df['description'] = df.apply(create_symptom_description, axis=1)

### SECTION TWO: making the training and testing data
## Converting the prognosis column into numbers to make it easier for the model to interpret
## have to prepare the dataset to fit the transformer format that the hugging face tool uses
## train test split

import sklearn
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

# Convert 'prognosis' to numerical labels
label_encoder = LabelEncoder()
df['prognosis_label'] = label_encoder.fit_transform(df['prognosis'])

# Prepare the dataset for Hugging Face's format
data_dict = {
    'text': df['description'].tolist(),
    'label': df['prognosis_label'].tolist()
}
dataset = Dataset.from_dict(data_dict)

# Split the dataset into training and testing sets
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

test_dataset.to_csv('test_dataset.csv', index=False)