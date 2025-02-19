# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

# Streamlit App title
st.title("Terrorism Attack Success Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Check if file is uploaded
if uploaded_file is not None:
    # Read CSV file into dataframe
    terr_lr_new = pd.read_csv(uploaded_file)
    
    # Display first few rows of the dataset
    st.write("Dataset Preview:")
    st.write(terr_lr_new.head())

    # Preprocessing
    terr_lr_new = terr_lr_new.drop(['Unnamed: 0'], axis=1)  # Dropping unnamed columns

    # Target variable
    y_lr = terr_lr_new['success']

    # Defining the features
    X_lr = terr_lr_new.drop(['success'], axis=1)

    # One-hot encoding
    X_lr_1 = pd.get_dummies(X_lr, columns=['attacktype1_txt', 'targtype1_txt', 'gname', 'weaptype1_txt'], dtype=int)

    # Clean column names
    X_lr_1.columns = X_lr_1.columns.str.replace('[^A-Za-z0-9_()]+', '_', regex=True)

    # Split into training and testing
    X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_lr_1, y_lr, test_size=0.2, random_state=10)

    # Scaling the data
    scaler = StandardScaler()
    X_lr_train_scaled = scaler.fit_transform(X_lr_train)
    X_lr_test_scaled = scaler.transform(X_lr_test)

    # Build and train the Logistic Regression model
    logR = LogisticRegression(random_state=10)
    logR.fit(X_lr_train_scaled, y_lr_train)

    # Predicting the model
    y_lr_pred = logR.predict(X_lr_test_scaled)

    # Displaying prediction results
    st.write(f"Predictions: {y_lr_pred[:10]}")  # Show first 10 predictions

    # Calculating probabilities
    y_lr_pred_prob = logR.predict_proba(X_lr_test_scaled)

    # Confusion Matrix
    st.write("Confusion Matrix:")
    conf_matrix = confusion_matrix(y_lr_test, y_lr_pred)
    st.write(conf_matrix)

    # Classification Report
    st.write("Classification Report:")
    class_report = classification_report(y_lr_test, y_lr_pred)
    st.text(class_report)

    # ROC Curve
    st.write("ROC Curve:")
    fpr, tpr, threshold = roc_curve(y_lr_test, y_lr_pred_prob[:, 1])
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    st.pyplot(fig)

    # ROC AUC Score
    st.write(f"ROC AUC Score: {roc_auc_score(y_lr_test, y_lr_pred)}")

    # Cross-validation
    st.write("Cross-validation (5-fold) Accuracy:")
    scores = cross_val_score(logR, X_lr_train, y_lr_train, cv=5, scoring='accuracy')
    st.write(f"Cross-Validation Scores: {scores}")
    st.write(f"Mean Accuracy: {scores.mean()}")

    # Cross-validated predictions
    st.write("Cross-validated Predictions:")
    y_pred = cross_val_predict(logR, X_lr_train, y_lr_train, cv=5)
    st.write(f"Predicted Values: {y_pred[:10]}")  # Show first 10 cross-validated predictions

    # Accuracy score
    st.write(f"Accuracy Score: {accuracy_score(y_lr_test, y_lr_pred)}")

