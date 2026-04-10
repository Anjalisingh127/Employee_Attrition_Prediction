# File: Employee_Attrition_Prediction-main/create_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# ❌ Removed PCA import (not needed anymore)
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline  # Crucial for SMOTE
from imblearn.over_sampling import SMOTE
import joblib  # For saving the pipeline
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

print("Script started: Building the deployment pipeline...")

# --- 1. Load Data ---
DATASET_NAME = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'

try:
    data = pd.read_csv(DATASET_NAME)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: Dataset file not found.")
    print(f"Please make sure '{DATASET_NAME}' is in the same folder as this script.")
    exit()

# --- 2. Define Features (X) and Target (y) ---
data = data.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'])

y = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
X = data.drop('Attrition', axis=1)

print("Features (X) and target (y) have been separated.")

# --- 3. Identify Categorical and Numerical Features ---
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

print(f"Found {len(numerical_features)} numerical features.")
print(f"Found {len(categorical_features)} categorical features.")

# --- 4. Create Preprocessing Transformers ---
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# --- 5. Create the ColumnTransformer ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# --- 6. Create the Full Prediction Pipeline ---
# ✅ PCA REMOVED → SHAP will now work correctly

full_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),                  # Step 1: Clean & transform data
    ('smote', SMOTE(random_state=42)),              # Step 2: Oversample (training only)
    ('model', RandomForestClassifier(n_estimators=100, random_state=42)) # Step 3: Predict
])

print("Full pipeline (preprocess, SMOTE, model) created.")

# --- 7. Train the Pipeline ---
print("Starting pipeline training...")
full_pipeline.fit(X, y)
print("Pipeline training complete.")

# --- 8. Save the Pipeline ---
FILE_NAME = 'attrition_pipeline.pkl'
joblib.dump(full_pipeline, FILE_NAME)

print("---------------------------------------------------------")
print(f"✅ Success! Pipeline saved as '{FILE_NAME}'")
print("This file contains your complete, trained model.")
print("---------------------------------------------------------")