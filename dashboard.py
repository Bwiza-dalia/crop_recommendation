import streamlit as st
import pandas as pd
from datetime import datetime

# Load the crop data into a DataFrame
crop_data = pd.read_csv('sample_dataset2.xlsx - Sheet1.csv')

def calculate_suitability_score(row):
    """
    Calculate the suitability score for a crop based on various factors.
    """
    score = 0
    score += row['Altitude (masl)'] * 0.2
    score += row['Annual rainfall (mm)'] * 0.15
    score += row['Crop water need (mm/total growing period)'] * 0.1
    score += row['Growing period (days)'] * 0.15
    score += row['Humidity_max'] * 0.1
    score += row['Temperature_max'] * 0.1
    score += row['Moisture_max'] * 0.1
    score += row['potassium_max'] * 0.05
    score += row['phosphorus_max'] * 0.05
    return score

def get_top_crop_recommendations(sector_name, district_name, start_date):
    """
    Recommends the top 3 crops based on the user's input.
    """
    # Get the sector and district IDs from their names
    sector_id = crop_data[crop_data['Sector'] == sector_name]['Sect_ID'].iloc[0]
    district_id = crop_data[crop_data['District'] == district_name]['Dist_ID'].iloc[0]
    
    # Filter the data based on the user's inputs
    filtered_data = crop_data[(crop_data['Sect_ID'] == sector_id) & 
                             (crop_data['Dist_ID'] == district_id)]
    
    # Get the month name from the start date
    month_name = datetime.strptime(start_date, '%Y-%m-%d').strftime('%B')
    start_month_col = f"Crop calendar start (month)_{month_name}"
    
    if start_month_col in crop_data.columns:
        filtered_data = filtered_data[filtered_data[start_month_col] == True]
    else:
        filtered_data = filtered_data[filtered_data['Crop calendar start (month)_Year-round'] == True]
    
    # Calculate the suitability score for each crop
    filtered_data['suitability_score'] = filtered_data.apply(calculate_suitability_score, axis=1)
    
    # Sort the filtered data by the suitability score and get the top 3 recommendations
    top_recommendations = filtered_data.sort_values('suitability_score', ascending=False)['Crop'].head(3).tolist()
    
    return top_recommendations

# Set up the Streamlit app
st.set_page_config(page_title="Crop Recommendation System")
st.title("Crop Recommendation System")

# Get unique sector and district names
sectors = sorted(crop_data['Sector'].unique())
districts = sorted(crop_data['District'].unique())

# Get user inputs
sector = st.selectbox("Select Sector", options=sectors)
district = st.selectbox("Select District", options=districts)
start_date = st.date_input("Select Start Date")

# Add a button to get recommendations
if st.button("Get Recommendations"):
    # Get the top recommendations
    top_crops = get_top_crop_recommendations(sector, district, str(start_date))
    
    # Display the results
    st.subheader("Top 3 Recommended Crops:")
    if top_crops:
        for i, crop in enumerate(top_crops, 1):
            st.write(f"{i}. {crop}")
    else:
        st.write("No recommendations found for the selected criteria.")