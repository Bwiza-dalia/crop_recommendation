import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Load the crop data into a DataFrame
crop_data = pd.read_csv('sample_dataset2.xlsx - Sheet1.csv')

def calculate_suitability_score(row, location_data):
    """
    Calculate the suitability score for a crop based on various factors and location conditions.
    """
    score = 0
    
    # Location-specific matching
    altitude_match = abs(1 - abs(row['Altitude (masl)'] - location_data['Altitude (masl)']) / location_data['Altitude (masl)']) * 20
    rainfall_match = abs(1 - abs(row['Annual rainfall (mm)'] - location_data['Annual rainfall (mm)']) / location_data['Annual rainfall (mm)']) * 15
    humidity_match = abs(1 - abs(row['Humidity_max'] - location_data['Humidity']) / location_data['Humidity']) * 15
    temperature_match = abs(1 - abs(row['Temperature_max'] - location_data['Temperature']) / location_data['Temperature']) * 15
    
    # Add location-specific scores
    score += altitude_match
    score += rainfall_match
    score += humidity_match
    score += temperature_match
    
    # Crop characteristics scores
    score += (row['Crop water need (mm/total growing period)'] / 1000) * 10  # Normalized water need
    score += (row['Growing period (days)'] / 365) * 10  # Normalized growing period
    score += (row['Moisture_max'] / 100) * 5  # Normalized moisture
    score += (row['potassium_max'] / 100) * 5  # Normalized potassium content
    score += (row['phosphorus_max'] / 100) * 5  # Normalized phosphorus content
    
    # Penalty for irrigation requirement if high
    if row['Irrigation required(%)'] > 50:
        score -= 10
    
    return max(0, min(100, score))  # Ensure score is between 0 and 100

def get_top_crop_recommendations(sector_name, district_name, start_date):
    """
    Recommends the top 5 crops based on the user's input.
    """
    try:
        # Get the sector and district IDs from their names
        sector_id = crop_data[crop_data['Sector'] == sector_name]['Sect_ID'].iloc[0]
        district_id = crop_data[crop_data['District'] == district_name]['Dist_ID'].iloc[0]
        
        # Get location-specific data (average values for the selected sector and district)
        location_data = {
            'Altitude (masl)': crop_data[(crop_data['Sect_ID'] == sector_id) & 
                                       (crop_data['Dist_ID'] == district_id)]['Altitude (masl)'].mean(),
            'Annual rainfall (mm)': crop_data[(crop_data['Sect_ID'] == sector_id) & 
                                            (crop_data['Dist_ID'] == district_id)]['Annual rainfall (mm)'].mean(),
            'Humidity': crop_data[(crop_data['Sect_ID'] == sector_id) & 
                                (crop_data['Dist_ID'] == district_id)]['Humidity'].mean(),
            'Temperature': crop_data[(crop_data['Sect_ID'] == sector_id) & 
                                   (crop_data['Dist_ID'] == district_id)]['Temperature'].mean()
        }
        
        # Filter the data based on the user's inputs
        filtered_data = crop_data[(crop_data['Sect_ID'] == sector_id) & 
                                 (crop_data['Dist_ID'] == district_id)].copy()
        
        # Get the month name from the start date
        month_name = datetime.strptime(start_date, '%Y-%m-%d').strftime('%B')
        start_month_col = f"Crop calendar start (month)_{month_name}"
        
        # Filter by planting season
        if start_month_col in filtered_data.columns:
            season_data = filtered_data[filtered_data[start_month_col] == True]
            if len(season_data) > 0:
                filtered_data = season_data
        
        # Calculate suitability scores for remaining crops
        filtered_data['suitability_score'] = filtered_data.apply(
            lambda row: calculate_suitability_score(row, location_data), axis=1
        )
        
        # Sort and get top 5 recommendations with scores
        top_recommendations = (filtered_data.sort_values('suitability_score', ascending=False)
                             .head(5)[['Crop', 'suitability_score']])
        
        return list(zip(top_recommendations['Crop'], 
                       top_recommendations['suitability_score']))
        
    except Exception as e:
        st.error(f"Error in recommendation generation: {str(e)}")
        return []

# Set up the Streamlit app
st.set_page_config(page_title="Crop Recommendation System")
st.title("Crop Recommendation System")

# Get unique sector and district names
sectors = sorted(crop_data['Sector'].unique())
districts = sorted(crop_data['District'].unique())

# Get user inputs
district = st.selectbox("Select District", options=districts)
sector = st.selectbox("Select Sector", options=sectors)
start_date = st.date_input("Select Start Date")

# Add a button to get recommendations
if st.button("Get Recommendations"):
    # Get the top recommendations
    recommendations = get_top_crop_recommendations(sector, district, str(start_date))
    
    # Display the results
    st.subheader("Top 5 Recommended Crops:")
    if recommendations:
        for i, (crop, score) in enumerate(recommendations, 1):
            st.write(f"{i}. {crop} (Suitability Score: {score:.2f}%)")
            
        # Display recommendation explanation
        st.subheader("Why these recommendations?")
        st.write("""
        The recommendations are based on:
        - Location altitude suitability
        - Local rainfall patterns
        - Temperature requirements
        - Humidity conditions
        - Growing season compatibility
        - Crop water needs
        - Soil conditions (moisture, potassium, phosphorus)
        - Irrigation requirements
        """)
    else:
        st.write("No recommendations found for the selected criteria.")