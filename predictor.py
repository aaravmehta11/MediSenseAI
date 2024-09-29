import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder

health_advice = {
    "Fungal Infection": "1. Follow prescribed antifungal medications. \n2. Keep the affected area clean and dry. \n3. Avoid tight clothing and moisture.",
    "Allergy": "1. Identify and avoid triggers. \n2. Use antihistamines as prescribed. \n3. Consult an allergist for testing and treatment options.",
    "GERD": "1. Avoid spicy and fatty foods. \n2. Eat smaller, more frequent meals. \n3. Elevate the head while sleeping.",
    "Chronic Cholestasis": "1. Follow a low-fat diet. \n2. Avoid alcohol and drugs that can worsen liver function. \n3. Regular follow-ups with a healthcare provider.",
    "Drug Reaction": "1. Discontinue the offending medication immediately. \n2. Inform your healthcare provider about the reaction. \n3. Seek emergency care if symptoms are severe.",
    "Peptic Ulcer Disease": "1. Avoid NSAIDs, alcohol, and smoking. \n2. Follow a bland diet; avoid spicy foods. \n3. Take prescribed medications regularly.",
    "AIDS": "1. Adhere to antiretroviral therapy (ART). \n2. Maintain a healthy lifestyle with proper nutrition and exercise. \n3. Regular medical check-ups and screenings.",
    "Diabetes": "1. Monitor blood sugar levels regularly. \n2. Follow a balanced diet with controlled carbohydrate intake. \n3. Exercise regularly and maintain a healthy weight.",
    "Gastroenteritis": "1. Stay hydrated with fluids and electrolytes. \n2. Avoid solid foods until nausea subsides. \n3. Rest as much as possible.",
    "Bronchial Asthma": "1. Avoid triggers such as smoke and allergens. \n2. Use inhalers as prescribed. \n3. Have an asthma action plan in place.",
    "Hypertension": "1. Follow a low-sodium diet. \n2. Exercise regularly. \n3. Take prescribed medications consistently.",
    "Migraine": "1. Identify and avoid triggers. \n2. Maintain a regular sleep schedule. \n3. Consult a doctor for medication options.",
    "Cervical Spondylosis": "1. Practice good posture. \n2. Engage in physical therapy. \n3. Consider pain management strategies.",
    "Paralysis (brain hemorrhage)": "1. Seek immediate medical attention. \n2. Engage in rehabilitation therapy. \n3. Follow-up care with specialists.",
    "Jaundice": "1. Avoid alcohol and fatty foods. \n2. Stay hydrated. \n3. Follow-up care for underlying liver issues.",
    "Malaria": "1. Seek immediate medical treatment. \n2. Follow prescribed antimalarial medications. \n3. Prevent mosquito bites with nets and repellents.",
    "Chickenpox": "1. Stay home to avoid spreading the virus. \n2. Take antihistamines to relieve itching. \n3. Consult a doctor for severe cases.",
    "Dengue": "1. Stay hydrated. \n2. Use pain relievers like acetaminophen. \n3. Seek medical attention for severe symptoms.",
    "Typhoid": "1. Follow a prescribed antibiotic treatment. \n2. Stay hydrated and maintain a proper diet. \n3. Practice good hygiene to prevent spread.",
    "Hepatitis A": "1. Avoid alcohol and certain medications. \n2. Maintain a nutritious diet. \n3. Get vaccinated for Hepatitis A.",
    "Hepatitis B": "1. Follow medical advice and treatment. \n2. Avoid alcohol and certain medications. \n3. Get vaccinated for Hepatitis B.",
    "Hepatitis C": "1. Follow prescribed antiviral therapy. \n2. Avoid alcohol and certain medications. \n3. Regular follow-ups with a healthcare provider.",
    "Hepatitis D": "1. Follow medical advice and treatment. \n2. Avoid alcohol and certain medications. \n3. Get vaccinated for Hepatitis B.",
    "Hepatitis E": "1. Stay hydrated and maintain a nutritious diet. \n2. Avoid alcohol and certain medications. \n3. Practice good hygiene.",
    "Alcoholic Hepatitis": "1. Avoid alcohol completely. \n2. Follow a healthy diet. \n3. Seek support for alcohol cessation.",
    "Tuberculosis": "1. Follow prescribed antibiotic treatment. \n2. Avoid close contact with others. \n3. Complete the full course of treatment.",
    "Common Cold": "1. Rest and hydrate. \n2. Use over-the-counter medications for symptom relief. \n3. Practice good hygiene to prevent spread.",
    "Pneumonia": "1. Seek medical attention. \n2. Follow prescribed antibiotics or antivirals. \n3. Rest and stay hydrated.",
    "Dimorphic Hemorrhoids (piles)": "1. Increase fiber intake to prevent constipation. \n2. Stay hydrated. \n3. Consider over-the-counter treatments for relief.",
    "Heart Attack": "1. Call emergency services immediately. \n2. Chew an aspirin if not allergic. \n3. Stay calm and try to remain still.",
    "Varicose Veins": "1. Elevate the legs to reduce swelling. \n2. Wear compression stockings. \n3. Consider medical treatment options.",
    "Hypothyroidism": "1. Follow prescribed thyroid hormone replacement therapy. \n2. Maintain a balanced diet. \n3. Regular monitoring of thyroid levels.",
    "Hyperthyroidism": "1. Follow prescribed treatment options. \n2. Maintain a balanced diet. \n3. Regular monitoring of thyroid levels.",
    "Hypoglycemia": "1. Eat small, frequent meals. \n2. Carry a fast-acting source of glucose. \n3. Monitor blood sugar levels regularly.",
    "Osteoarthritis": "1. Engage in low-impact exercise. \n2. Maintain a healthy weight. \n3. Consider pain management options.",
    "Arthritis": "1. Engage in regular exercise. \n2. Follow prescribed medication regimens. \n3. Consider physical therapy.",
    "Vertigo": "1. Avoid sudden movements. \n2. Stay hydrated. \n3. Consult a healthcare provider for treatment options.",
    "Acne": "1. Maintain a consistent skincare routine. \n2. Avoid touching the face. \n3. Consult a dermatologist for treatment options.",
    "Urinary Tract Infection": "1. Stay hydrated and urinate frequently. \n2. Take prescribed antibiotics. \n3. Avoid irritants like caffeine and alcohol.",
    "Psoriasis": "1. Follow prescribed topical treatments. \n2. Avoid known triggers. \n3. Maintain a healthy lifestyle.",
    "Impetigo": "1. Keep the affected area clean. \n2. Avoid scratching and picking. \n3. Follow prescribed antibiotic treatment."
}

# Load your fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained("./my_finetuned_model")
tokenizer = BertTokenizer.from_pretrained("./my_finetuned_model")

# Load the original prognosis data (assuming you have access to it)
df = pd.read_csv("symbipredict_2022.csv")
label_encoder = LabelEncoder()
label_encoder.fit(df['prognosis'])  # Fit the encoder with the original prognosis data

# Define a function for prediction
def predict_prognosis(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    predicted_prognosis = label_encoder.inverse_transform([predicted_class_id])[0]
    advice = health_advice.get(predicted_prognosis, "No additional advice available.")
    
    return predicted_prognosis, advice


### STREAMLIT STUFF ###

# Streamlit UI
st.title("Medical Symptom Checker")

# Create a text area for user input
user_input = st.text_area("Enter your symptoms. Be as detailed as possible to ensure a more accurate prognosis.", "e.g., Chest pain, shortness of breath")

# Predict prognosis when the user clicks the button
if st.button("Predict Prognosis"):
    if user_input:
        result, advice = predict_prognosis(user_input)
        st.success(f"Predicted Prognosis: {result}")
        st.write("### Health Advice:")
        st.write(advice)
    else:
        st.error("Please enter some symptoms.")