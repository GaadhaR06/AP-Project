import pandas as pd
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Load the pre-trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')  # Load feature names used during training
    return model, scaler, feature_names

# Streamlit App title
st.title("Terrorism Attack Success Prediction")

# Upload CSV file (without the 'success' column)
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    terr_lr_new = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the dataset
    st.write("Dataset Preview (without 'success' column):")
    st.write(terr_lr_new.head())

    # Check for missing values in the uploaded file
    if terr_lr_new.isnull().sum().any():
        st.warning(f"Missing values in the dataset: {terr_lr_new.isnull().sum()}")

    # Load the pre-trained model, scaler, and feature names
    model, scaler, trained_feature_names = load_model()

    # Preprocess the uploaded data (this should match the preprocessing steps done during training)
    # Drop the 'success' column if it exists (to avoid any user mistakes)
    if 'success' in terr_lr_new.columns:
        terr_lr_new = terr_lr_new.drop(['success'], axis=1)
    
    # One-hot encode categorical features (same as in the training process)
    X_new = pd.get_dummies(terr_lr_new, columns=['attacktype1_txt', 'targtype1_txt', 'gname', 'weaptype1_txt'], dtype=int)

    # Clean the column names
    X_new.columns = X_new.columns.str.replace('[^A-Za-z0-9_()]+', '_', regex=True)

    # Reindex the columns to match the trained model's columns (ensure no features are missing)
    missing_cols = set(trained_feature_names) - set(X_new.columns)
    for col in missing_cols:
        X_new[col] = 0

    # Reorder columns to match the trained model's feature order
    X_new = X_new[trained_feature_names]

    # Scale the features using the same scaler that was used for training
    X_new_scaled = scaler.transform(X_new)

    # Predict the 'success' using the pre-trained model
    y_pred = model.predict(X_new_scaled)
    y_pred_prob = model.predict_proba(X_new_scaled)[:, 1]  # Probability of class 1 (success)

    # Add the predictions to the DataFrame
    terr_lr_new['predicted_success'] = y_pred
    terr_lr_new['predicted_success_prob'] = y_pred_prob

    # Display the predictions 
    st.write("Predicted Results (Success or Failure):")
    st.write(terr_lr_new[['predicted_success', 'predicted_success_prob']])

    # # Plot the ROC curve
    # fpr, tpr, threshold = roc_curve(y_pred, y_pred_prob)
    # fig, ax = plt.subplots()
    # ax.plot(fpr, tpr)
    # ax.set_xlabel('False Positive Rate')
    # ax.set_ylabel('True Positive Rate')
    # ax.set_title('ROC Curve')
    # st.pyplot(fig)

    # # Calculate and display ROC AUC score
    # roc_auc = roc_auc_score(y_pred, y_pred_prob)
    # st.write(f"ROC AUC Score: {roc_auc}")
