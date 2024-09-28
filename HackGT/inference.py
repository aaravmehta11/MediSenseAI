### The inteference tests the model by loading it in to run predictions with it ###
from transformers import BertForSequenceClassification, BertTokenizer 
from sklearn.preprocessing import LabelEncoder
import pandas as pd

model = BertForSequenceClassification.from_pretrained("./my_finetuned_model")
tokenizer = BertTokenizer.from_pretrained("./my_finetuned_model")

# Load the original prognosis data (assuming you have access to it)
df = pd.read_csv("symbipredict_2022.csv")
label_encoder = LabelEncoder()
label_encoder.fit(df['prognosis'])  # Fit the encoder with the original prognosis data

# Inference: predicting prognosis for new user input
def predict_prognosis(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    predicted_prognosis = label_encoder.inverse_transform([predicted_class_id])[0]
    return predicted_prognosis

# Example input
user_input = "I have pain in my anal region and have constipation."

print(predict_prognosis(user_input))