# File: app.py (FINAL VERSION - SHAP ERROR FIXED + DASHBOARD CONNECTED)

import streamlit as st
import pandas as pd
import joblib
import warnings
import os
import base64
import re
import shap
import numpy as np   # ✅ Added

# ✅ ADD THIS (for BA dashboard connection)
import sys
sys.path.append(os.path.dirname(__file__))
import ba_dashboard

# Gmail API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

warnings.filterwarnings('ignore')

# ================= LOAD MODELS =================

@st.cache_resource
def load_attrition_pipeline():
    return joblib.load("../Employee_Attrition_Prediction-main/attrition_pipeline.pkl")

@st.cache_resource
def load_nlp_model():
    return joblib.load('nlp_model.pkl')

attrition_pipeline = load_attrition_pipeline()
nlp_pipeline = load_nlp_model()

# ================= SHAP =================

model = attrition_pipeline.named_steps['model']
preprocessor = attrition_pipeline.named_steps['preprocessor']

@st.cache_resource
def load_shap_explainer():
    try:
        sample_data = pd.DataFrame([{
            'Age': 35, 'BusinessTravel': 'Travel_Rarely', 'DailyRate': 800,
            'Department': 'Sales', 'DistanceFromHome': 5, 'Education': 3,
            'EducationField': 'Life Sciences', 'EnvironmentSatisfaction': 3,
            'Gender': 'Male', 'HourlyRate': 65, 'JobInvolvement': 3,
            'JobLevel': 2, 'JobRole': 'Sales Executive', 'JobSatisfaction': 3,
            'MaritalStatus': 'Single', 'MonthlyIncome': 5000,
            'MonthlyRate': 14000, 'NumCompaniesWorked': 2,
            'OverTime': 'No', 'PercentSalaryHike': 15,
            'PerformanceRating': 3, 'RelationshipSatisfaction': 3,
            'StockOptionLevel': 1, 'TotalWorkingYears': 10,
            'TrainingTimesLastYear': 3, 'WorkLifeBalance': 3,
            'YearsAtCompany': 5, 'YearsInCurrentRole': 4,
            'YearsSinceLastPromotion': 2, 'YearsWithCurrManager': 4
        }])

        background = preprocessor.transform(sample_data)
        return shap.TreeExplainer(model, background)

    except Exception:
        return shap.TreeExplainer(model)

explainer = load_shap_explainer()

# ================= GMAIL =================

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

@st.cache_data
def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

#def clean_email_body(text):
#     if not text:
#         return ""
#     try:
#         text = base64.urlsafe_b64decode(text).decode('utf-8')
#     except:
#         pass
#     text = re.sub(r'<[^>]+>', ' ', text)
#     text = re.sub(r'http\S+|www\S+', ' ', text)
#     text = re.sub(r'\S+@\S+', ' ', text)
#     return re.sub(r'\s+', ' ', text).strip()

def clean_email_body(text):
    if not text:
        return ""

    try:
        text = base64.urlsafe_b64decode(text).decode('utf-8', errors='ignore')
    except:
        pass

    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[\u200c\u200d\u200e\u200f]', '', text)
    text = re.sub(r'[^a-zA-Z0-9.,!? ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def fetch_unread_emails(service):
    results = service.users().messages().list(userId='me', labelIds=['UNREAD'], maxResults=10).execute()
    messages = results.get('messages', [])

    emails = []
    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        headers = msg_data['payload']['headers']

        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), '')

        payload = msg_data['payload']
        body = ""

        if 'parts' in payload:
            body = payload['parts'][0]['body'].get('data', '')
        else:
            body = payload['body'].get('data', '')

        emails.append({
            'sender': sender,
            'subject': subject,
            'body': clean_email_body(body)
        })

    return emails

# ================= UI =================

st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")
st.sidebar.title("HR Dashboard")

# ✅ UPDATED NAVIGATION (only change here)
page = st.sidebar.radio("Select Tool", [
    "Attrition Prediction",
    "Email Scanner",
    "Business Analytics Dashboard"
])

# =====================================================
# ATTRITION PREDICTION
# =====================================================

if page == "Attrition Prediction":

    st.title("Employee Attrition Prediction (SHAP Enabled)")

    st.write("DEBUG PIPELINE ")
    st.write(attrition_pipeline)

    def user_input_features():

        return pd.DataFrame({
            'Age': [st.slider("Age", 18, 60, 35)],
            'BusinessTravel': [st.selectbox("Business Travel", ("Non-Travel", "Travel_Rarely", "Travel_Frequently"))],
            'DailyRate': [st.slider("Daily Rate", 100, 1500, 800)],
            'Department': [st.selectbox("Department", ("Sales", "Research & Development", "Human Resources"))],
            'DistanceFromHome': [st.slider("Distance From Home", 1, 30, 5)],
            'Education': [st.selectbox("Education", [1,2,3,4,5])],
            'EducationField': [st.selectbox("Education Field", ("Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"))],
            'EnvironmentSatisfaction': [st.slider("Environment Satisfaction", 1, 4, 3)],
            'Gender': [st.selectbox("Gender", ("Male", "Female"))],
            'HourlyRate': [st.slider("Hourly Rate", 30, 100, 65)],
            'JobInvolvement': [st.slider("Job Involvement", 1, 4, 3)],
            'JobLevel': [st.slider("Job Level", 1, 5, 2)],
            'JobRole': [st.selectbox("Job Role", (
                "Sales Executive","Research Scientist","Laboratory Technician",
                "Manufacturing Director","Healthcare Representative","Manager",
                "Sales Representative","Research Director","Human Resources"
            ))],
            'JobSatisfaction': [st.slider("Job Satisfaction", 1, 4, 3)],
            'MaritalStatus': [st.selectbox("Marital Status", ("Married", "Single", "Divorced"))],
            'MonthlyIncome': [st.slider("Monthly Income", 1000, 20000, 5000)],
            'MonthlyRate': [st.slider("Monthly Rate", 2000, 27000, 14000)],
            'NumCompaniesWorked': [st.slider("Companies Worked", 0, 9, 2)],
            'OverTime': [st.selectbox("OverTime", ("Yes", "No"))],
            'PercentSalaryHike': [st.slider("Percent Salary Hike", 10, 25, 15)],
            'PerformanceRating': [st.slider("Performance Rating", 1, 4, 3)],
            'RelationshipSatisfaction': [st.slider("Relationship Satisfaction", 1, 4, 3)],
            'StockOptionLevel': [st.slider("Stock Option Level", 0, 3, 1)],
            'TotalWorkingYears': [st.slider("Total Working Years", 0, 40, 10)],
            'TrainingTimesLastYear': [st.slider("Training Times Last Year", 0, 6, 3)],
            'WorkLifeBalance': [st.slider("Work Life Balance", 1, 4, 3)],
            'YearsAtCompany': [st.slider("Years At Company", 0, 40, 5)],
            'YearsInCurrentRole': [st.slider("Years In Current Role", 0, 18, 4)],
            'YearsSinceLastPromotion': [st.slider("Years Since Last Promotion", 0, 15, 2)],
            'YearsWithCurrManager': [st.slider("Years With Current Manager", 0, 17, 4)]
        })

    input_df = user_input_features()
    st.write("### Input Data", input_df)

    if st.button("Predict"):

        pred = attrition_pipeline.predict(input_df)[0]
        prob = attrition_pipeline.predict_proba(input_df)[0][1] * 100

        if pred == 1:
            st.error(f"High Attrition Risk ({prob:.2f}%)")
        else:
            st.success(f"Low Attrition Risk ({prob:.2f}%)")

        # ================= SHAP =================
        st.markdown("---")
        st.subheader("Explainable AI (SHAP)")

        try:
            X_transformed = preprocessor.transform(input_df)
            shap_values = explainer.shap_values(X_transformed)

            if isinstance(shap_values, list):
                shap_vals = shap_values[1]
            else:
                shap_vals = shap_values

            shap_vals = np.array(shap_vals)

            if shap_vals.ndim > 1:
                shap_vals = shap_vals[0]

            shap_vals = shap_vals.flatten()

            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                feature_names = [f"Feature_{i}" for i in range(len(shap_vals))]

            min_len = min(len(feature_names), len(shap_vals))
            feature_names = feature_names[:min_len]
            shap_vals = shap_vals[:min_len]

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Impact": shap_vals
            })

            shap_df = shap_df.reindex(shap_df.Impact.abs().sort_values(ascending=False).index)

            st.bar_chart(shap_df.head(10).set_index("Feature"))

            st.write("Top Factors Influencing Prediction")
            top_features = shap_df.head(5)

            for _, row in top_features.iterrows():
                direction = "increasing" if row["Impact"] > 0 else "decreasing"
                st.write(f"{row['Feature']} is {direction} attrition risk")

        except Exception as e:
            st.error(f"SHAP error: {e}")

# =====================================================
# EMAIL SCANNER
# =====================================================

elif page == "Email Scanner":

    st.title("Real-Time Email Scanner")

    if st.button("Scan Inbox"):

        service = get_gmail_service()
        emails = fetch_unread_emails(service)

        vectorizer = nlp_pipeline.steps[0][1]
        classifier = nlp_pipeline.steps[1][1]

        feature_names = vectorizer.get_feature_names_out()
        weights = classifier.coef_[0]

        for i, email in enumerate(emails):

            text = email['body']
            pred = nlp_pipeline.predict([text])[0]
            prob = nlp_pipeline.predict_proba([text])[0][pred]

            with st.expander(f"{email['subject']} - {email['sender']}"):

                if pred == 1:
                    st.error(f"Resignation Intent ({prob*100:.2f}%)")

                    words = text.lower().split()

                    df = pd.DataFrame({
                        "word": feature_names,
                        "weight": weights
                    })

                    df = df[df.word.isin(words)].nlargest(5, 'weight')

                    if not df.empty:
                        st.table(df)
                    else:
                        st.write("Complex pattern detected")

                else:
                    st.success(f"Normal Email ({prob*100:.2f}%)")

                st.text_area("Email Body", text, height=150, key=i)

# =====================================================
# BA DASHBOARD
# =====================================================

elif page == "Business Analytics Dashboard":
    #ba_dashboard.run()
    ba_dashboard.run_ba_dashboard()