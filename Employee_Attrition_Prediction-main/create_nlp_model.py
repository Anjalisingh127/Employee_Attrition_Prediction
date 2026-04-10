# File: Employee_Attrition_Prediction-main/create_nlp_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

print("Script started: Building the NLP email model...")

# --- 1. Load Data ---
DATASET_NAME = 'email_dataset.csv'

try:
    data = pd.read_csv(DATASET_NAME)
    print("Email dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: Dataset file not found.")
    print(f"Please make sure '{DATASET_NAME}' is in the same folder as this script.")
    exit()

# --- 2. Define Features (X) and Target (y) ---
X = data['text']
y = data['label']

print("Features (X) and target (y) separated.")

# --- 3. Create the NLP Pipeline ---
# This pipeline will do two things:
# 1. TfidfVectorizer: Convert text into a matrix of numbers.
# 2. LogisticRegression: The model that will classify the text.
nlp_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=500)),
    ('model', LogisticRegression(random_state=42))
])

print("NLP pipeline (TF-IDF + Logistic Regression) created.")

# --- 4. Train the Model ---
# We can use a simple train/test split to see how well it works
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training the NLP model...")
nlp_pipeline.fit(X_train, y_train)
print("Training complete.")

# --- 5. Evaluate the Model (Optional, but good practice) ---
print("\n--- Model Performance on Test Data ---")
y_pred = nlp_pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Label 0 (Normal)', 'Label 1 (High-Risk)']))
print("-------------------------------------------\n")

# --- 6. Save the Trained Pipeline ---
# ! IMPORTANT: We save this directly into the 'attrition_app' folder
# This path assumes 'attrition_app' is one level *up* and then *down*
SAVE_PATH = '../attrition_app/nlp_model.pkl'

try:
    joblib.dump(nlp_pipeline, SAVE_PATH)
    print(f"✅ Success! NLP model saved to: {SAVE_PATH}")
except Exception as e:
    print(f"Error saving model: {e}")
    print("Please ensure the 'attrition_app' folder exists at this location:")
    print("...Employee_Attrition_Prediction-main/attrition_app/")

print("---------------------------------------------------------")