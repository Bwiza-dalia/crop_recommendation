import pandas as pd
from datetime import datetime

# Load the crop data into a DataFrame
crop_data = pd.read_csv('sample_dataset2.xlsx - Sheet1.csv')

def calculate_suitability_score(row):
    """
    Calculate the suitability score for a crop based on various factors.
    
    Parameters:
    row (pandas.Series): A row of the crop data DataFrame
    
    Returns:
    float: The calculated suitability score
    """
    # Example suitability score calculation based on some factors
    # You can modify this calculation based on your requirements
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

def get_top_crop_recommendations(sector, district, start_date):
    """
    Recommends the top 3 crops based on the user's input of sector, district, and start date.
    
    Parameters:
    sector (str): The sector the user is interested in
    district (str): The district the user is located in
    start_date (str): The start date of the crop season in the format 'YYYY-MM-DD'
    
    Returns:
    list: A list of the top 3 recommended crop types
    """
    # Filter the data based on the user's inputs
    filtered_data = crop_data[(crop_data['Sect_ID'] == sector) & 
                              (crop_data['Dist_ID'] == district)]
    
    # Check if the start date matches any of the available start month columns
    start_month_col = f"Crop calendar start (month)_{start_date.split('-')[1]}"
    if start_month_col in crop_data.columns:
        filtered_data = filtered_data[filtered_data[start_month_col] == True]
    else:
        # If the start date doesn't match any of the available start month columns, check the "Year-round" column
        filtered_data = filtered_data[filtered_data['Crop calendar start (month)_Year-round'] == True]
    
    # Calculate the suitability score for each crop
    filtered_data['suitability_score'] = filtered_data.apply(calculate_suitability_score, axis=1)
    
    # Sort the filtered data by the suitability score and get the top 3 recommendations
    top_recommendations = filtered_data.sort_values('suitability_score', ascending=False)['Crop'].head(3).tolist()
    
    return top_recommendations