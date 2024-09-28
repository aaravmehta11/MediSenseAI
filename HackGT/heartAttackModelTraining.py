### SECTION ONE: creating and filtering the dataset
## Converts the data columns of symptoms to one column with a String describing the symptoms of the patient
## ex: is chest_pain == 1 and coughing == 1 in the original dataset, then the new dataset will have "The patient has chest pain, coughing."

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

### SECTION THREE: BERT
## tokenize into bert format so that we can feed the model into bert
## convert the old training and testing dataset into ones that bert can interpret and understand
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)


### SECTION FOUR: TRAINING THE MODEL
## load in the normal bert model
## create the training arguments and run the trainer on it in order to fine tune the model
## save the fine-tuned 
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Get the number of unique prognoses (classes)
num_labels = len(label_encoder.classes_)

# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=num_labels
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory
    evaluation_strategy="epoch",     # evaluate at the end of each epoch
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    num_train_epochs=3,              # number of training epochs
    weight_decay=0.01,               # strength of weight decay
    logging_dir="./logs",            # directory for storing logs
)

# Define the trainer
trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=test_dataset
)

# Fine-tune the model
trainer.train()

import os

os.makedirs("./my_finetuned_model", exist_ok=True)
model.save_pretrained("./my_finetuned_model")
tokenizer.save_pretrained("./my_finetuned_model")

### NEW SECTION ###

# Evaluate the model on the test dataset
# trainer.evaluate()
