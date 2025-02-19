# app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer

# Load the saved model, encoder, and imputer
model = joblib.load('random_forest_model.pkl')
encoder = joblib.load('one_hot_encoder.pkl')
imputer = joblib.load('imputer.pkl')

# Streamlit web app
st.title("Random Forest Model Prediction")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file with a different encoding to avoid decoding errors
        data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Fill missing or blank 'nkill' values with 0 (just in case it's present in test data)
    if 'nkill' in data.columns:
        data['nkill'] = data['nkill'].apply(lambda x: 0 if pd.isnull(x) or x == '' else x)

    # Handle missing values for independent variables (imputation)
    X_test = data.drop(columns=['nkill'], errors='ignore')  # 'nkill' might not exist in test data

    # Impute missing values in independent variables
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # One-hot encode categorical variables
    categorical_columns = X_test_imputed.select_dtypes(include=['object']).columns
    X_test_encoded = pd.DataFrame(encoder.transform(X_test_imputed[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))

    # Drop the original categorical columns and combine with the encoded columns
    X_test_imputed = X_test_imputed.drop(columns=categorical_columns)
    X_test_final = pd.concat([X_test_imputed, X_test_encoded], axis=1)

    # Predict the 'nkill' category
    predictions = model.predict(X_test_final)

    # Map predictions to low, medium, high
    prediction_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    predictions_mapped = [prediction_map[pred] for pred in predictions]

    # Display the predictions
    st.write("Predictions for 'nkill' category:")
    data['nkill_pred'] = predictions_mapped
    st.write(data[['nkill_pred']])
