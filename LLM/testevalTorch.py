import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from datasets import Dataset

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained("./my_finetuned_model")
tokenizer = BertTokenizer.from_pretrained("./my_finetuned_model")

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load your test dataset (replace with your actual test dataset file)
df_test = pd.read_csv("tokenized_test_dataset.csv")

# Create a Dataset object compatible with Hugging Face's trainer
test_data_dict = {
    'text': df_test['text'].tolist(),   # The text column
    'label': df_test['label'].tolist()  # The numeric labels column
}
test_dataset = Dataset.from_dict(test_data_dict)

# Tokenize the test dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove unneeded columns for evaluation (keep only input_ids, attention_mask, and labels)
test_dataset = test_dataset.remove_columns(['text'])  
test_dataset.set_format("torch")

# Define Trainer object for evaluation (you don't need to provide a train dataset)
trainer = Trainer(
    model=model,
)

# Evaluate the model using the test dataset
predictions_output = trainer.predict(test_dataset)

# Get the predicted labels
predictions = torch.argmax(torch.tensor(predictions_output.predictions), dim=-1).numpy()

# Get the true labels from the test set
labels = predictions_output.label_ids

# Calculate metrics
accuracy = accuracy_score(labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

# Print metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
