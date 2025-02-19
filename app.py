import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, roc_auc_score
from folium.plugins import MarkerCluster

# Streamlit App title
st.title("Terrorism Analytics Web App")

# Sidebar for navigation
option = st.sidebar.selectbox(
    "Choose an Analysis Type",
    ("Random Forest Model Prediction", "Logistic Regression Model Prediction", "Geospatial Visualization")
)

# --- Random Forest Model Prediction ---
if option == "Random Forest Model Prediction":
    st.header("Random Forest Model Prediction")

    # Load the saved model, encoder, and imputer
    model = joblib.load('random_forest_model.pkl')
    encoder = joblib.load('one_hot_encoder.pkl')
    imputer = joblib.load('imputer.pkl')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file with a different encoding to avoid decoding errors
            data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()

        # Fill missing or blank 'nkill' values with 0
        if 'nkill' in data.columns:
            data['nkill'] = data['nkill'].apply(lambda x: 0 if pd.isnull(x) or x == '' else x)

        # Handle missing values for independent variables (imputation)
        X_test = data.drop(columns=['nkill'], errors='ignore')

        # Impute missing values in independent variables
        X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

        # One-hot encode categorical variables
        categorical_columns = X_test_imputed.select_dtypes(include=['object']).columns
        X_test_encoded = pd.DataFrame(encoder.transform(X_test_imputed[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))

        # Combine the imputed and encoded columns
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

# --- Logistic Regression Model Prediction ---
elif option == "Logistic Regression Model Prediction":
    st.header("Logistic Regression Model Prediction")

    # Load the pre-trained model and scaler
    @st.cache_resource
    def load_model():
        model = joblib.load('logistic_regression_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')  # Load feature names used during training
        return model, scaler, feature_names

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

# --- Geospatial Visualization ---
elif option == "Geospatial Visualization":
    st.header("Terrorism Casualties and Geospatial Visualization")

    # Upload Excel file
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

    if uploaded_file is not None:
        # Read the uploaded Excel file
        terr_ombd = pd.read_excel(uploaded_file)
        
        # Display the first few rows of the dataset
        st.write("Dataset Preview:")
        st.write(terr_ombd.head())

        # Clean the data
        terr_ombd['provstate'] = terr_ombd['provstate'].replace('Andhra pradesh', 'Andhra Pradesh')
        terr_ombd['provstate'] = terr_ombd['provstate'].replace('Orissa', 'Odisha')
        terr_ombd = terr_ombd[terr_ombd['provstate'] != 'Unknown']

        # Selecting the required columns
        terr_ombd = terr_ombd[['provstate', 'city', 'latitude', 'longitude', 'nkill', 'nwound']]

        # Combining necessary columns and filling missing values
        terr_ombd['casualities'] = terr_ombd['nkill'] + terr_ombd['nwound']
        terr_ombd.fillna(0, inplace=True)

        # Dropping old columns
        terr_ombd.drop(['nkill', 'nwound'], axis=1, inplace=True)

        # Remove rows where latitude or longitude is NaN
        terr_ombd = terr_ombd.dropna(subset=['latitude', 'longitude'])

        # Grouping by province and calculating weighted latitude and longitude
        result = terr_ombd.groupby("provstate").apply(lambda x: pd.Series({
            "OutputLatitude": (x["latitude"] * x["casualities"]).sum() / x["casualities"].sum(),
            "OutputLongitude": (x["longitude"] * x["casualities"]).sum() / x["casualities"].sum(),
            "casualities": x["casualities"].sum()
        })).reset_index()

        # Display cleaned data
        st.write("Processed Data:")
        st.write(result)

        # Create a map using Folium
        st.write("Map of Terrorism Casualties (Weighted Locations)")

        # Create a folium map centered on India
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

        # Add a MarkerCluster for better map management
        marker_cluster = MarkerCluster().add_to(m)

        # Add CircleMarkers to the map based on processed data
        for _, row in result.iterrows():
            # Ensure there are no NaN values for latitude and longitude before plotting
            if pd.notna(row['OutputLatitude']) and pd.notna(row['OutputLongitude']):
                folium.CircleMarker(
                    location=[row['OutputLatitude'], row['OutputLongitude']],
                    radius=row['casualities'] / 100,  # Scale radius for better visualization
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.6,
                    popup=f"{row['provstate']} - Casualties: {row['casualities']}"
                ).add_to(marker_cluster)

        # Display the map in Streamlit
        st.markdown("### Terrorism Casualties Map")
        st.components.v1.html(m._repr_html_(), height=500)

        # Optional: Display basic statistics
        st.write("Basic Statistics on Casualties by Province:")
        st.write(result.describe())

