import streamlit as st
import pandas as pd
import folium
import numpy as np
import matplotlib.pyplot as plt
from folium.plugins import MarkerCluster

# Streamlit App title
st.title("Terrorism Casualties and Geospatial Visualization")

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
    terr_ombd['provstate'] = terr_ombd['provstate'].replace('Orissa','Odisha')
    terr_ombd = terr_ombd[terr_ombd['provstate'] != 'Unknown']

    # Selecting the required columns
    terr_ombd = terr_ombd[['provstate', 'city', 'latitude', 'longitude', 'nkill', 'nwound']]

    # Combining necessary columns and filling missing values
    terr_ombd['casualities'] = terr_ombd['nkill'] + terr_ombd['nwound']
    terr_ombd.fillna(0, inplace=True)

    # Dropping old columns
    terr_ombd.drop(['nkill', 'nwound'], axis=1, inplace=True)

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

