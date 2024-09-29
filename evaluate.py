import torch
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained("./my_finetuned_model")
tokenizer = BertTokenizer.from_pretrained("./my_finetuned_model")


# Load the test dataset (already encoded and saved)
df_test = pd.read_csv("test_dataset.csv")

# Extract the text and labels from the test dataset
test_texts = df_test['text'].tolist()   # These are the symptom descriptions
test_labels = df_test['label'].tolist() # These are the numeric labels for prognosis

# Tokenize the input texts from the test dataset
def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

inputs = tokenize_function(test_texts)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ensure the model is in evaluation mode
model.eval()

# Disable gradient calculation for inference
with torch.no_grad():
    outputs = model(**inputs.to(device))
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()

# Calculate metrics
accuracy = accuracy_score(test_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='weighted')

# Print results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
