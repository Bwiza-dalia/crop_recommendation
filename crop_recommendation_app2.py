from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError


def clean_numeric_range(value):
    if isinstance(value, str):
        # If the value contains a range (e.g., '156-350'), take the average
        if '-' in value:
            low, high = map(float, value.split('-'))
            return (low + high) / 2
    return value

def clean_dataset(df):
    df_cleaned = df.copy()
    
    # List of numeric columns that might need cleaning
    numeric_columns = [
        'Altitude (masl)', 'Annual rainfall (mm)', 'N', 'P', 'K',
        'Humidity_min', 'Humidity_max', 'Temperature_min', 'Temperature_max',
        'Moisture_min', 'Moisture_max', 'pH_min', 'pH_max'
    ]
    
    # Clean each numeric column
    for col in numeric_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].apply(clean_numeric_range)
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    # Fill any NaN values with column medians
    df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].median())
    
    return df_cleaned


class CropRecommendationSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
    def preprocess_data(self, df_cleaned):
        """Preprocess the input data for model training"""
        # Select relevant features including environmental, soil, and seasonal factors
        self.relevant_features = [
            # Environmental conditions
            'Altitude (masl)'
            , 'Annual rainfall (mm)',
            'Temperature_min', 'Temperature_max', 
            'Humidity_min', 'Humidity_max',
            'Moisture_min', 'Moisture_max',
            
            # Soil nutrients
            'N', 'P', 'K',
            'pH_min', 'pH_max',
            'potassium_min', 'potassium_max',
            'phosphorus_min', 'phosphorus_max',
            
            # # Growing conditions
            'Crop water need (mm/total growing period)',
            'Growing period (days)',
            'Irrigation required(%)',
            
            # # Geographic features
            'Longitude', 'Latitude',
            
            # # Aggregate conditions
            'Humidity', 'Rainfall', 'Temperature', 
            'Elevation', 'Soil_Moisture',
            
            # # Calendar features - Start months
            'Crop calendar start (month)_April',
            'Crop calendar start (month)_August',
            'Crop calendar start (month)_February',
            'Crop calendar start (month)_January',
            'Crop calendar start (month)_July',
            'Crop calendar start (month)_November',
            'Crop calendar start (month)_October',
            'Crop calendar start (month)_September',
            'Crop calendar start (month)_Year-round',
            
            # # Calendar features - End months
            'Crop calendar end (month)_April',
            'Crop calendar end (month)_January',
            'Crop calendar end (month)_July',
            'Crop calendar end (month)_March',
            'Crop calendar end (month)_November',
            'Crop calendar end (month)_October',
            'Crop calendar end (month)_Year-round'
        ]
        
        # Create X (features) and y (target)
        X = df_cleaned[self.relevant_features].copy()
        y = df_cleaned['Crop']
        
        # Scale numerical features (excluding boolean calendar features)
        numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
        X_scaled = X.copy()
        X_scaled[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X_scaled, y_encoded
    
    def train_model(self, X, y):
        """Train the Random Forest model"""
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        
        return X_train, X_test, y_train, y_test, y_pred
    
    def get_feature_importance(self):
        """Get the importance of each feature in making predictions"""
        feature_importance = pd.DataFrame({
            'feature': self.relevant_features,
            'importance': self.model.feature_importances_
        })
        return feature_importance.sort_values('importance', ascending=False)
    
    def recommend_crop(self, input_data):
        """Recommend crop based on input conditions"""
        # Ensure all required features are present
        missing_features = set(self.relevant_features) - set(input_data.columns)
        if missing_features:
            for feature in missing_features:
                if feature.startswith('Crop calendar'):
                    input_data[feature] = False
                else:
                    input_data[feature] = 0
                    
        # Scale numerical features
        numerical_features = input_data.select_dtypes(include=['float64', 'int64']).columns
        input_scaled = input_data.copy()
        input_scaled[numerical_features] = self.scaler.transform(input_data[numerical_features])
        
        # Ensure correct feature order
        input_scaled = input_scaled[self.relevant_features]
        
        # Get predictions and probabilities
        probabilities = self.model.predict_proba(input_scaled)
        
        # Get top 3 crop recommendations with probabilities
        top_3_idx = np.argsort(probabilities[0])[-3:][::-1]
        recommendations = []
        
        for idx in top_3_idx:
            crop = self.label_encoder.inverse_transform([idx])[0]
            probability = probabilities[0][idx]
            recommendations.append({
                'crop': crop,
                'confidence': probability
            })
            
        return recommendations

def main():
    st.set_page_config(page_title="Crop Recommendation System")
    st.title("Crop Recommendation System")

    # Load and preprocess the data
    df = pd.read_csv('sample_dataset2.xlsx - Sheet1.csv')
    df_cleaned = clean_dataset(df)
    X, y = recommender.preprocess_data(df_cleaned)

    # Train the model
    X_train, X_test, y_train, y_test, y_pred = recommender.train_model(X, y)

    # Collect user inputs
    st.header("Input Conditions")
    user_inputs = {
        'Altitude (masl)': st.number_input("Altitude (masl)", min_value=0, step=1),
        'Annual rainfall (mm)': st.number_input("Annual Rainfall (mm)", min_value=0, step=1),
        'Temperature_min': st.number_input("Minimum Temperature (°C)", min_value=-50.0, max_value=50.0, step=0.1),
        'Temperature_max': st.number_input("Maximum Temperature (°C)", min_value=-50.0, max_value=50.0, step=0.1),
        'Humidity_min': st.number_input("Minimum Humidity (%)", min_value=0.0, max_value=100.0, step=0.1),
        'Humidity_max': st.number_input("Maximum Humidity (%)", min_value=0.0, max_value=100.0, step=0.1),
        'Moisture_min': st.number_input("Minimum Soil Moisture (%)", min_value=0.0, max_value=100.0, step=0.1),
        'Moisture_max': st.number_input("Maximum Soil Moisture (%)", min_value=0.0, max_value=100.0, step=0.1),
        'N': st.number_input("Nitrogen (N)", min_value=0.0, step=0.1),
        'P': st.number_input("Phosphorus (P)", min_value=0.0, step=0.1),
        'K': st.number_input("Potassium (K)", min_value=0.0, step=0.1),
        'pH_min': st.number_input("Minimum pH", min_value=0.0, max_value=14.0, step=0.1),
        'pH_max': st.number_input("Maximum pH", min_value=0.0, max_value=14.0, step=0.1),
    }

    # Add calendar feature inputs (default to False if missing)
    for month_feature in [
        'Crop calendar start (month)_April', 'Crop calendar start (month)_August', 'Crop calendar start (month)_February', 
        'Crop calendar start (month)_January', 'Crop calendar start (month)_July', 'Crop calendar start (month)_November', 
        'Crop calendar start (month)_October', 'Crop calendar start (month)_September', 'Crop calendar start (month)_Year-round',
        'Crop calendar end (month)_April', 'Crop calendar end (month)_January', 'Crop calendar end (month)_July', 
        'Crop calendar end (month)_March', 'Crop calendar end (month)_November', 'Crop calendar end (month)_October', 
        'Crop calendar end (month)_Year-round'
    ]:
        user_inputs[month_feature] = st.checkbox(month_feature, value=False)

    if st.button("Get Recommendations"):
        # Prepare the input DataFrame in the exact order of `relevant_features`
        input_data = pd.DataFrame({feature: [user_inputs.get(feature, 0)] for feature in recommender.relevant_features})

        try:
            # Get the recommendations
            recommendations = recommender.recommend_crop(input_data)

            # Display the recommendations
            st.header("Top 3 Recommended Crops:")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec['crop']} (Confidence: {rec['confidence']:.2%})")
        except NotFittedError:
            st.error("The model is not fitted yet. Please check if the training process ran successfully.")

if __name__ == "__main__":
    recommender = CropRecommendationSystem()
    main()