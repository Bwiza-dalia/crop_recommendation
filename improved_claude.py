import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.exceptions import NotFittedError
import joblib
import os
import folium
from streamlit_folium import folium_static
import calendar
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point
import geopandas as gpd
from folium.plugins import MarkerCluster


def clean_numeric_range(value):
    """Clean numeric values that might be in range format"""
    if isinstance(value, str):
        # If the value contains a range (e.g., '156-350'), take the average
        if '-' in value:
            try:
                low, high = map(float, value.split('-'))
                return (low + high) / 2
            except ValueError:
                return np.nan
        # Try to convert other string formats to float
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value


def clean_dataset(df):
    """Clean and preprocess the dataset"""
    df_cleaned = df.copy()
    
    # List of numeric columns that might need cleaning
    numeric_columns = [
        'Altitude (masl)', 'Annual rainfall (mm)', 'N', 'P', 'K',
        'Humidity_min', 'Humidity_max', 'Temperature_min', 'Temperature_max',
        'Moisture_min', 'Moisture_max', 'pH_min', 'pH_max',
        'potassium_min', 'potassium_max', 'phosphorus_min', 'phosphorus_max',
        'Crop water need (mm/total growing period)', 'Growing period (days)',
        'Irrigation required(%)', 'Longitude', 'Latitude', 
        'Humidity', 'Rainfall', 'Temperature', 'Elevation', 'Soil_Moisture'
    ]
    
    # Clean each numeric column if it exists
    for col in numeric_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].apply(clean_numeric_range)
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    # Fill NaN values with column medians for numeric columns
    for col in numeric_columns:
        if col in df_cleaned.columns and df_cleaned[col].isna().any():
            median_value = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_value)
    
    return df_cleaned


class CropRecommendationSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,  # Limit depth to prevent overfitting
            min_samples_split=5,  # Require more samples to split a node
            min_samples_leaf=2,  # Require more samples in leaf nodes
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.relevant_features = []
        self.model_trained = False
        self.crops = []
        self.feature_importance_df = None
        
        # For location-based recommendations
        self.location_data = None
        self.soil_data = None
        self.climate_data = None
        
    def preprocess_data(self, df_cleaned):
        """Preprocess the input data for model training"""
        # Store crop names
        self.crops = df_cleaned['Crop'].unique().tolist()
        
        # Dynamically determine available features in the dataset
        # List of potential relevant features to check
        potential_features = [
            # Environmental conditions
            'Altitude (masl)', 'Annual rainfall (mm)',
            'Temperature_min', 'Temperature_max', 
            'Humidity_min', 'Humidity_max',
            'Moisture_min', 'Moisture_max',
            
            # Soil nutrients
            'N', 'P', 'K',
            'pH_min', 'pH_max',
            'potassium_min', 'potassium_max',
            'phosphorus_min', 'phosphorus_max',
            
            # Growing conditions
            'Crop water need (mm/total growing period)',
            'Growing period (days)',
            'Irrigation required(%)',
            
            # Geographic features
            'Longitude', 'Latitude',
            
            # Aggregate conditions
            'Humidity', 'Rainfall', 'Temperature', 
            'Elevation', 'Soil_Moisture',
        ]
        
        # Find available features in the dataset
        self.relevant_features = [f for f in potential_features if f in df_cleaned.columns]
        
        # Add calendar features if present
        calendar_features = [col for col in df_cleaned.columns 
                            if 'Crop calendar' in col and col != 'Crop calendar']
        self.relevant_features.extend(calendar_features)
        
        if not self.relevant_features:
            raise ValueError("No relevant features found in the dataset")
        
        # Create X (features) and y (target)
        X = df_cleaned[self.relevant_features].copy()
        
        if 'Crop' not in df_cleaned.columns:
            raise ValueError("Target column 'Crop' not found in the dataset")
            
        y = df_cleaned['Crop']
        
        # Scale numerical features
        numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
        X_scaled = X.copy()
        
        if not numerical_features.empty:
            X_scaled[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create location-based mappings if geographical coordinates are available
        if 'Longitude' in self.relevant_features and 'Latitude' in self.relevant_features:
            self.location_data = df_cleaned[['Longitude', 'Latitude', 'Crop']].copy()
            
            # If soil and climate data are available, store them for recommendations
            soil_cols = [col for col in df_cleaned.columns if col in ['N', 'P', 'K', 'pH_min', 'pH_max']]
            if soil_cols:
                self.soil_data = df_cleaned[['Longitude', 'Latitude'] + soil_cols].copy()
                
            climate_cols = [col for col in df_cleaned.columns if col in [
                'Temperature_min', 'Temperature_max', 'Humidity_min', 'Humidity_max',
                'Annual rainfall (mm)', 'Moisture_min', 'Moisture_max'
            ]]
            if climate_cols:
                self.climate_data = df_cleaned[['Longitude', 'Latitude'] + climate_cols].copy()
        
        return X_scaled, y_encoded
    
    def train_model(self, X, y):
        """Train the Random Forest model with cross-validation"""
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy and F1 score
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Perform cross-validation to get a more realistic accuracy estimate
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.model, X, y, cv=cv, scoring='accuracy'
        )
        
        # Calculate feature importance
        self.feature_importance_df = self.get_feature_importance()
        
        self.model_trained = True
        
        return {
            'X_train': X_train, 
            'X_test': X_test, 
            'y_train': y_train, 
            'y_test': y_test, 
            'y_pred': y_pred, 
            'accuracy': accuracy,
            'f1_score': f1,
            'cv_scores': cv_scores
        }
    
    def get_feature_importance(self):
        """Get the importance of each feature in making predictions"""
        if not hasattr(self.model, 'feature_importances_'):
            return None
            
        feature_importance = pd.DataFrame({
            'feature': self.relevant_features,
            'importance': self.model.feature_importances_
        })
        return feature_importance.sort_values('importance', ascending=False)
    
    def recommend_crop(self, input_data):
        """Recommend crop based on input conditions"""
        if not self.model_trained:
            raise NotFittedError("The model has not been trained yet")
            
        # Ensure all required features are present
        missing_features = set(self.relevant_features) - set(input_data.columns)
        if missing_features:
            for feature in missing_features:
                if 'Crop calendar' in feature:
                    input_data[feature] = False
                else:
                    input_data[feature] = 0
                    
        # Scale numerical features
        numerical_features = input_data.select_dtypes(include=['float64', 'int64']).columns
        input_scaled = input_data.copy()
        
        if not numerical_features.empty:
            try:
                input_scaled[numerical_features] = self.scaler.transform(input_data[numerical_features])
            except ValueError as e:
                st.error(f"Error scaling input data: {e}")
                return []
        
        # Ensure correct feature order
        input_scaled = input_scaled[self.relevant_features]
        
        try:
            # Get predictions and probabilities
            probabilities = self.model.predict_proba(input_scaled)
            
            # Get top crop recommendations with probabilities
            top_idx = np.argsort(probabilities[0])[::-1]
            recommendations = []
            
            for idx in top_idx[:5]:  # Get top 5 recommendations
                crop = self.label_encoder.inverse_transform([idx])[0]
                probability = probabilities[0][idx]
                recommendations.append({
                    'crop': crop,
                    'confidence': probability
                })
                
            return recommendations
        except Exception as e:
            st.error(f"Error making predictions: {e}")
            return []
    
    def get_location_based_recommendations(self, longitude, latitude):
        """Get crop recommendations based on nearest locations in the training data"""
        if self.location_data is None:
            return None
        
        # Calculate distance to all points in the dataset
        self.location_data['distance'] = np.sqrt(
            (self.location_data['Longitude'] - longitude)**2 + 
            (self.location_data['Latitude'] - latitude)**2
        )
        
        # Get the 5 nearest neighbors
        nearest = self.location_data.sort_values('distance').head(5)
        
        # Get crop frequencies from nearest neighbors
        crop_counts = nearest['Crop'].value_counts().reset_index()
        crop_counts.columns = ['crop', 'count']
        total = crop_counts['count'].sum()
        crop_counts['confidence'] = crop_counts['count'] / total
        
        # Format as recommendations
        recommendations = []
        for _, row in crop_counts.iterrows():
            recommendations.append({
                'crop': row['crop'],
                'confidence': row['confidence']
            })
            
        return recommendations
    
    def get_seasonal_recommendations(self, longitude, latitude, start_month):
        """Get crop recommendations based on location and planting season"""
        # First get location-based recommendations
        base_recommendations = self.get_location_based_recommendations(longitude, latitude)
        
        if not base_recommendations:
            return None
            
        # Filter recommended crops by those suitable for the selected planting month
        seasonal_recs = []
        
        # Check if we have calendar data
        calendar_features = [f for f in self.relevant_features if 'Crop calendar start' in f]
        
        if calendar_features:
            # Get crops that can be planted in the selected month
            month_feature = f"Crop calendar start (month)_{calendar.month_name[start_month]}"
            
            # If we have the specific month feature
            if month_feature in self.relevant_features:
                # Adjust confidence scores - boost those that match the season
                for rec in base_recommendations:
                    crop_name = rec['crop']
                    
                    # This is simplified - in a real implementation you'd check your dataset
                    if self.is_crop_suitable_for_month(crop_name, start_month):
                        seasonal_recs.append({
                            'crop': crop_name,
                            'confidence': rec['confidence'] * 1.2  # Boost confidence
                        })
                    else:
                        seasonal_recs.append({
                            'crop': crop_name,
                            'confidence': rec['confidence'] * 0.8  # Reduce confidence
                        })
                
                # Sort by adjusted confidence
                seasonal_recs = sorted(seasonal_recs, key=lambda x: x['confidence'], reverse=True)
                
                # Normalize confidences to sum to 1
                total_conf = sum(rec['confidence'] for rec in seasonal_recs)
                for rec in seasonal_recs:
                    rec['confidence'] = rec['confidence'] / total_conf
                
                return seasonal_recs
        
        # If we don't have calendar data, return the base recommendations
        return base_recommendations
    
    def is_crop_suitable_for_month(self, crop, month):
        """Check if a crop is suitable for planting in a given month"""
        # This would ideally reference your dataset's growing season information
        # Simplified implementation - would need to be customized based on your data
        month_name = calendar.month_name[month]
        
        # Check if we have crops with specific calendar data
        df_with_crop = pd.DataFrame({
            'Crop': self.crops,
            f'Crop calendar start (month)_{month_name}': [
                f'Crop calendar start (month)_{month_name}' in self.relevant_features
                for _ in self.crops
            ]
        })
        
        return True  # Simplified - always return True if we don't have the data
    
    def get_nearest_soil_data(self, longitude, latitude):
        """Get soil characteristics based on nearest locations in the training data"""
        if self.soil_data is None:
            return None
        
        # Calculate distance to all points in the dataset
        self.soil_data['distance'] = np.sqrt(
            (self.soil_data['Longitude'] - longitude)**2 + 
            (self.soil_data['Latitude'] - latitude)**2
        )
        
        # Get the nearest point
        nearest = self.soil_data.sort_values('distance').iloc[0].copy()
        
        # Remove distance column
        if 'distance' in nearest:
            nearest = nearest.drop('distance')
            
        # Remove location columns
        if 'Longitude' in nearest:
            nearest = nearest.drop('Longitude')
        if 'Latitude' in nearest:
            nearest = nearest.drop('Latitude')
            
        return nearest
    
    def get_nearest_climate_data(self, longitude, latitude):
        """Get climate characteristics based on nearest locations in the training data"""
        if self.climate_data is None:
            return None
        
        # Calculate distance to all points in the dataset
        self.climate_data['distance'] = np.sqrt(
            (self.climate_data['Longitude'] - longitude)**2 + 
            (self.climate_data['Latitude'] - latitude)**2
        )
        
        # Get the nearest point
        nearest = self.climate_data.sort_values('distance').iloc[0].copy()
        
        # Remove distance column
        if 'distance' in nearest:
            nearest = nearest.drop('distance')
            
        # Remove location columns
        if 'Longitude' in nearest:
            nearest = nearest.drop('Longitude')
        if 'Latitude' in nearest:
            nearest = nearest.drop('Latitude')
            
        return nearest
    
    def save_model(self, path='crop_recommendation_model.joblib'):
        """Save the trained model to disk"""
        if not self.model_trained:
            return False
            
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'relevant_features': self.relevant_features,
                'crops': self.crops,
                'feature_importance': self.feature_importance_df,
                'location_data': self.location_data,
                'soil_data': self.soil_data,
                'climate_data': self.climate_data
            }
            joblib.dump(model_data, path)
            return True
        except Exception as e:
            return False
    
    def load_model(self, path='crop_recommendation_model.joblib'):
        """Load a trained model from disk"""
        try:
            if os.path.exists(path):
                model_data = joblib.load(path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.label_encoder = model_data['label_encoder']
                self.relevant_features = model_data['relevant_features']
                
                if 'crops' in model_data:
                    self.crops = model_data['crops']
                if 'feature_importance' in model_data:
                    self.feature_importance_df = model_data['feature_importance']
                if 'location_data' in model_data:
                    self.location_data = model_data['location_data']
                if 'soil_data' in model_data:
                    self.soil_data = model_data['soil_data']
                if 'climate_data' in model_data:
                    self.climate_data = model_data['climate_data']
                    
                self.model_trained = True
                return True
            else:
                return False
        except Exception as e:
            return False


def plot_feature_importance(feature_importance_df, top_n=10):
    """Plot feature importance using Plotly"""
    if feature_importance_df is None or feature_importance_df.empty:
        return None
        
    # Get top N features
    top_features = feature_importance_df.head(top_n)
    
    # Create Plotly horizontal bar chart
    fig = px.bar(
        top_features, 
        x='importance', 
        y='feature',
        orientation='h',
        title=f'Top {top_n} Most Important Features',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Viridis'
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=400,
        width=700
    )
    
    return fig


def plot_crop_distribution(df):
    """Plot distribution of crops in the dataset"""
    if df is None or 'Crop' not in df.columns:
        return None
    
    # Get crop counts
    crop_counts = df['Crop'].value_counts().reset_index()
    crop_counts.columns = ['Crop', 'Count']
    
    # Create pie chart using Plotly
    fig = px.pie(
        crop_counts, 
        values='Count', 
        names='Crop', 
        title='Distribution of Crops in Dataset',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    # Customize layout
    fig.update_layout(
        height=400,
        width=700
    )
    
    return fig


def plot_correlation_matrix(df, features):
    """Plot correlation matrix for selected features"""
    if df is None or not all(feature in df.columns for feature in features):
        return None
    
    if len(features) < 2:
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    
    # Create heatmap using Plotly
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title='Correlation Matrix of Key Features',
        aspect='auto'
    )
    
    # Customize layout
    fig.update_layout(
        height=500,
        width=700
    )
    
    return fig


def plot_crop_distribution(df):
    """Plot distribution of crops in the dataset"""
    if df is None or 'Crop' not in df.columns:
        return None
    
    # Get crop counts
    crop_counts = df['Crop'].value_counts().reset_index()
    crop_counts.columns = ['Crop', 'Count']
    
    # Create pie chart using Plotly
    fig = px.pie(
        crop_counts, 
        values='Count', 
        names='Crop', 
        title='Distribution of Crops in Dataset',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    # Customize layout
    fig.update_layout(
        height=400,
        width=700
    )
    
    return fig


def plot_soil_characteristics(df):
    """Plot soil characteristics by crop"""
    if df is None or not all(col in df.columns for col in ['Crop', 'N', 'P', 'K', 'pH_min', 'pH_max']):
        return None
    
    # Group by crop and calculate mean values
    soil_by_crop = df.groupby('Crop')[['N', 'P', 'K', 'pH_min', 'pH_max']].mean().reset_index()
    
    # Create radar chart for each crop
    fig = go.Figure()
    
    for crop in soil_by_crop['Crop'].unique():
        crop_data = soil_by_crop[soil_by_crop['Crop'] == crop]
        
        fig.add_trace(go.Scatterpolar(
            r=[
                crop_data['N'].values[0],
                crop_data['P'].values[0],
                crop_data['K'].values[0],
                crop_data['pH_min'].values[0],
                crop_data['pH_max'].values[0],
            ],
            theta=['Nitrogen', 'Phosphorus', 'Potassium', 'pH min', 'pH max'],
            fill='toself',
            name=crop
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, soil_by_crop[['N', 'P', 'K', 'pH_min', 'pH_max']].max().max() * 1.1]
            )
        ),
        title='Soil Characteristics by Crop',
        height=500,
        width=700
    )
    
    return fig


def plot_climate_characteristics(df):
    """Plot climate characteristics by crop"""
    climate_features = [
        'Temperature_min', 'Temperature_max', 
        'Humidity_min', 'Humidity_max',
        'Annual rainfall (mm)'
    ]
    
    if df is None or not all(feature in df.columns for feature in climate_features):
        return None
    
    # Group by crop and calculate mean values
    climate_by_crop = df.groupby('Crop')[climate_features].mean().reset_index()
    
    # Create boxplot for temperature
    temp_fig = px.box(
        df,
        x='Crop',
        y=['Temperature_min', 'Temperature_max'],
        title='Temperature Range by Crop',
        color='Crop'
    )
    
    # Create boxplot for humidity
    humidity_fig = px.box(
        df,
        x='Crop',
        y=['Humidity_min', 'Humidity_max'],
        title='Humidity Range by Crop',
        color='Crop'
    )
    
    # Create boxplot for rainfall
    rainfall_fig = px.box(
        df,
        x='Crop',
        y='Annual rainfall (mm)',
        title='Annual Rainfall by Crop',
        color='Crop'
    )
    
    return temp_fig, humidity_fig, rainfall_fig


def plot_crop_map(df):
    """Plot crops on a map using folium"""
    if df is None or not all(col in df.columns for col in ['Crop', 'Latitude', 'Longitude']):
        return None
    
    # Create map centered on the mean coordinates
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add markers for each location
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Crop: {row['Crop']}",
            tooltip=row['Crop'],
            icon=folium.Icon(color=get_color_for_crop(row['Crop']))
        ).add_to(marker_cluster)
    
    return m


def get_color_for_crop(crop):
    """Get color for a crop (for map markers)"""
    # Simplified - in a real implementation you'd map specific crops to colors
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
              'darkpurple', 'pink', 'lightblue', 'lightgreen']
    
    # Hash the crop name to get a consistent color
    return colors[hash(crop) % len(colors)]


def farmer_friendly_interface():
    """Main function for the farmer-friendly interface"""
    st.set_page_config(
        page_title="Farmer's Crop Recommendation System",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS to make the interface look nicer
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1565C0;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-header'>🌱 Farmer's Crop Recommendation System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    This system helps you determine which crops are most suitable for your farm based on your location and 
    when you plan to start cultivation. Our recommendations take into account soil conditions, climate patterns, 
    and historical crop performance in your area.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize recommender system
    recommender = CropRecommendationSystem()
    
    # Create sidebar for data operations and expert mode
    with st.sidebar:
        st.title("System Controls")
        
        # Mode selector - Farmer mode or Expert mode
        mode = st.radio(
            "Select Mode",
            ["Farmer Mode", "Expert Mode"],
            index=0
        )
        
        if mode == "Expert Mode":
            st.header("Data Operations")
            
            # File uploader for expert mode
            uploaded_file = st.file_uploader(
                "Upload dataset (CSV or Excel)",
                type=["csv", "xlsx", "xls"]
            )
            
            model_file = 'crop_recommendation_model.joblib'
            model_exists = os.path.exists(model_file)
            
            # Model options for expert mode
            model_options = st.radio(
                "Model Options",
                ["Train new model", "Load existing model"],
                index=1 if model_exists else 0
            )
            
            # Save model button for expert mode
            save_model_button = st.button("Save Model")
        else:
            st.info("You're in Farmer Mode. Just select your location and planting month to get recommendations!")
    
    # Initialize variables
    df = None
    df_cleaned = None
    model_loaded = False
    center_lat = 0.0
    center_lon = 0.0
    
    # Expert Mode - Load or train model
    if mode == "Expert Mode":
        # Load existing model if selected
        if model_options == "Load existing model":
            with st.spinner("Loading existing model..."):
                if recommender.load_model():
                    model_loaded = True
                    st.success("Model loaded successfully!")
                    # Get center coordinates if location data is available
                    if recommender.location_data is not None:
                        center_lat = recommender.location_data['Latitude'].mean()
                        center_lon = recommender.location_data['Longitude'].mean()
        
        # Dataset loading and preprocessing
        if uploaded_file is not None:
            try:
                # Load data based on file extension
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:  # Excel file
                    df = pd.read_excel(uploaded_file)
                
                # Display dataset info
                st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records", len(df))
                with col2:
                    st.metric("Features", len(df.columns))
                with col3:
                    if 'Crop' in df.columns:
                        st.metric("Unique Crops", df['Crop'].nunique())
                
                # Clean dataset
                with st.spinner("Cleaning and preprocessing dataset..."):
                    df_cleaned = clean_dataset(df)
                
                # Data Exploration
                st.markdown("<h2 class='sub-header'>Data Exploration</h2>", unsafe_allow_html=True)
                
                # Show tabs for different visualizations
                viz_tabs = st.tabs([
                    "Crop Distribution", 
                    "Feature Correlation", 
                    "Soil Analysis", 
                    "Climate Analysis",
                    "Geographic Distribution"
                ])
                
                with viz_tabs[0]:  # Crop Distribution
                    crop_dist_fig = plot_crop_distribution(df_cleaned)
                    if crop_dist_fig:
                        st.plotly_chart(crop_dist_fig)
                    else:
                        st.warning("Couldn't generate crop distribution chart. Make sure the dataset has a 'Crop' column.")
                
                with viz_tabs[1]:  # Feature Correlation
                    # Select features for correlation matrix
                    corr_features = [col for col in df_cleaned.columns 
                                     if col in [
                                         'N', 'P', 'K', 'pH_min', 'pH_max',
                                         'Temperature_min', 'Temperature_max',
                                         'Humidity_min', 'Humidity_max',
                                         'Annual rainfall (mm)', 'Rainfall',
                                         'Moisture_min', 'Moisture_max'
                                     ] and col in df_cleaned.columns]
                                     
                    if corr_features:
                        corr_fig = plot_correlation_matrix(df_cleaned, corr_features)
                        if corr_fig:
                            st.plotly_chart(corr_fig)
                        else:
                            st.warning("Couldn't generate correlation matrix.")
                    else:
                        st.warning("No suitable numerical features found for correlation analysis.")
                
                with viz_tabs[2]:  # Soil Analysis
                    soil_fig = plot_soil_characteristics(df_cleaned)
                    if soil_fig:
                        st.plotly_chart(soil_fig)
                    else:
                        st.warning("Couldn't generate soil characteristics chart. Make sure the dataset has soil-related columns (N, P, K, pH).")
                
                with viz_tabs[3]:  # Climate Analysis
                    climate_figs = plot_climate_characteristics(df_cleaned)
                    if climate_figs:
                        for fig in climate_figs:
                            st.plotly_chart(fig)
                    else:
                        st.warning("Couldn't generate climate charts. Make sure the dataset has climate-related columns.")
                
                with viz_tabs[4]:  # Geographic Distribution
                    if 'Latitude' in df_cleaned.columns and 'Longitude' in df_cleaned.columns:
                        st.subheader("Geographic Distribution of Crops")
                        crop_map = plot_crop_map(df_cleaned)
                        if crop_map:
                            folium_static(crop_map)
                            # Set center coordinates for farmer mode
                            center_lat = df_cleaned['Latitude'].mean()
                            center_lon = df_cleaned['Longitude'].mean()
                        else:
                            st.warning("Couldn't generate crop map.")
                    else:
                        st.warning("Geographic coordinates (Latitude, Longitude) not found in the dataset.")
                
                # Train new model if selected
                if model_options == "Train new model":
                    st.markdown("<h2 class='sub-header'>Model Training</h2>", unsafe_allow_html=True)
                    
                    with st.spinner("Training model, please wait..."):
                        try:
                            # Preprocess data
                            X, y = recommender.preprocess_data(df_cleaned)
                            
                            # Train model
                            results = recommender.train_model(X, y)
                            
                            # Display model performance
                            st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Test Accuracy", f"{results['accuracy']:.2%}")
                            with col2:
                                st.metric("F1 Score", f"{results['f1_score']:.2%}")
                            with col3:
                                cv_mean = np.mean(results['cv_scores'])
                                st.metric("Cross-Val Accuracy", f"{cv_mean:.2%}")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            if cv_mean > 0.95:
                                st.markdown("""
                                <div class='warning-box'>
                                <strong>Note on High Accuracy:</strong> The model shows very high accuracy, which might 
                                indicate potential overfitting or data leakage. Consider the following:
                                <ul>
                                    <li>Ensure train/test data are properly separated</li>
                                    <li>Check for duplicate or highly correlated features</li>
                                    <li>Cross-validation provides a more realistic estimate of model performance</li>
                                </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Show confusion matrix
                            st.subheader("Confusion Matrix")
                            cm = confusion_matrix(results['y_test'], results['y_pred'])
                            class_names = recommender.label_encoder.classes_
                            
                            cm_fig = px.imshow(
                                cm,
                                x=class_names,
                                y=class_names,
                                text_auto=True,
                                color_continuous_scale='Blues',
                                title='Confusion Matrix',
                                labels=dict(x="Predicted", y="True", color="Count")
                            )
                            
                            st.plotly_chart(cm_fig)
                            
                            # Show feature importance
                            st.subheader("Feature Importance")
                            imp_fig = plot_feature_importance(recommender.feature_importance_df)
                            if imp_fig:
                                st.plotly_chart(imp_fig)
                            
                            model_loaded = True
                            
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")
                
                # Save model if button clicked
                if save_model_button and recommender.model_trained:
                    if recommender.save_model():
                        st.sidebar.success("Model saved successfully!")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Main interface - for both Farmer Mode and Expert Mode
    if model_loaded or mode == "Farmer Mode":
        # For Farmer Mode, always try to load the existing model
        if mode == "Farmer Mode" and not model_loaded:
            if not recommender.load_model():
                st.error("""
                No pre-trained model found. Please ask the system administrator to:
                1. Switch to Expert Mode
                2. Upload a dataset
                3. Train and save a model
                """)
                st.stop()
            else:
                model_loaded = True
                # Get center coordinates if location data is available
                if recommender.location_data is not None:
                    center_lat = recommender.location_data['Latitude'].mean()
                    center_lon = recommender.location_data['Longitude'].mean()
        
        # Create tabs for different recommendation methods
        st.markdown("<h2 class='sub-header'>Crop Recommendation</h2>", unsafe_allow_html=True)
        
        rec_tabs = st.tabs([
            "Location-Based (Simple)", 
            "Advanced Options"
        ])
        
        with rec_tabs[0]:  # Simple location-based recommendations
            st.markdown("""
            <div class='info-box'>
            Simply select your location on the map and when you plan to start planting. 
            We'll recommend suitable crops based on soil and climate patterns in your area.
            </div>
            """, unsafe_allow_html=True)
            
            # Create columns for map and inputs
            map_col, input_col = st.columns([2, 1])
            
            with map_col:
                # Check if we have location data
                if recommender.location_data is not None:
                    # Create map for location selection
                    st.subheader("Select Your Farm Location")
                    
                    # Create map with existing data points
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
                    
                    # Add cluster of existing points
                    marker_cluster = MarkerCluster().add_to(m)
                    
                    # Add markers for existing data
                    for _, row in recommender.location_data.iterrows():
                        folium.CircleMarker(
                            location=[row['Latitude'], row['Longitude']],
                            radius=5,
                            color=get_color_for_crop(row['Crop']),
                            fill=True,
                            fill_opacity=0.7,
                            popup=f"Crop: {row['Crop']}"
                        ).add_to(marker_cluster)
                    
                    # Display the map
                    folium_static(m, width=700)
                    
                    # Get clicked location (this would require additional JavaScript in a real implementation)
                    # Here we'll use a workaround with input fields
                    st.caption("Click on the map to select your location or enter coordinates manually below")
                else:
                    st.warning("No location data available. Please provide coordinates manually.")
            
            with input_col:
                st.subheader("Farm Details")
                
                # Location input
                loc_col1, loc_col2 = st.columns(2)
                
                with loc_col1:
                    latitude = st.number_input(
                        "Latitude",
                        min_value=-90.0,
                        max_value=90.0,
                        value=center_lat,
                        step=0.01
                    )
                
                with loc_col2:
                    longitude = st.number_input(
                        "Longitude",
                        min_value=-180.0,
                        max_value=180.0,
                        value=center_lon,
                        step=0.01
                    )
                
                # Planting month selection
                current_month = datetime.now().month
                start_month = st.selectbox(
                    "When do you plan to start planting?",
                    options=list(range(1, 13)),
                    format_func=lambda x: calendar.month_name[x],
                    index=current_month - 1
                )
                
                # Get recommendations button
                if st.button("Get Recommendations", type="primary", key="simple_rec_button"):
                    with st.spinner("Analyzing your farm location and generating recommendations..."):
                        # Get recommendations based on location and season
                        recommendations = recommender.get_seasonal_recommendations(
                            longitude, latitude, start_month
                        )
                        
                        if recommendations:
                            # Display recommendations
                            st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                            st.subheader("Recommended Crops for Your Farm")
                            
                            # Create columns for top recommendations
                            top_n = min(3, len(recommendations))
                            rec_cols = st.columns(top_n)
                            
                            for i, (col, rec) in enumerate(zip(rec_cols, recommendations[:top_n])):
                                col.metric(
                                    label=f"{i+1}. {rec['crop']}", 
                                    value=f"{rec['confidence']:.1%}"
                                )
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Display soil and climate data for the location if available
                            st.subheader("Your Farm Conditions (Estimated)")
                            
                            soil_data = recommender.get_nearest_soil_data(longitude, latitude)
                            climate_data = recommender.get_nearest_climate_data(longitude, latitude)
                            
                            if soil_data is not None or climate_data is not None:
                                cond_col1, cond_col2 = st.columns(2)
                                
                                with cond_col1:
                                    st.write("**Soil Conditions**")
                                    if soil_data is not None:
                                        for key, value in soil_data.items():
                                            st.write(f"{key}: {value:.2f}")
                                    else:
                                        st.write("No soil data available")
                                
                                with cond_col2:
                                    st.write("**Climate Conditions**")
                                    if climate_data is not None:
                                        for key, value in climate_data.items():
                                            st.write(f"{key}: {value:.2f}")
                                    else:
                                        st.write("No climate data available")
                            
                            # Display detailed bar chart of recommendations
                            st.subheader("Crop Suitability Scores")
                            chart_data = pd.DataFrame({
                                'Crop': [rec['crop'] for rec in recommendations],
                                'Suitability': [rec['confidence'] for rec in recommendations]
                            })
                            
                            rec_fig = px.bar(
                                chart_data, 
                                x='Crop', 
                                y='Suitability',
                                color='Suitability',
                                color_continuous_scale='Viridis',
                                title='Crop Suitability Scores for Your Farm',
                                text_auto='.1%'
                            )
                            
                            rec_fig.update_layout(
                                xaxis_title='Crop',
                                yaxis_title='Suitability Score',
                                yaxis=dict(range=[0, 1])
                            )
                            
                            st.plotly_chart(rec_fig)
                            
                            # Display planting tips
                            st.subheader("Planting Tips")
                            
                            for rec in recommendations[:3]:
                                with st.expander(f"Tips for {rec['crop']}"):
                                    st.write(f"""
                                    **Best Time to Plant:** {calendar.month_name[start_month]} to {calendar.month_name[(start_month + 2) % 12 or 12]}
                                    
                                    **General Requirements:**
                                    - Ensure adequate irrigation, especially during early growth
                                    - Monitor for pests and diseases regularly
                                    - Follow recommended spacing for optimal yield
                                    - Consider crop rotation to maintain soil health
                                    
                                    **Note:** These are general guidelines. Consult with local agricultural experts for specific advice for your area.
                                    """)
                        else:
                            st.error("Unable to generate recommendations for this location. Please try a different location or contact support.")
        
        with rec_tabs[1]:  # Advanced options
            st.markdown("""
            <div class='info-box'>
            For experienced farmers or those who know their soil conditions. 
            You can specify detailed environmental and soil parameters to get more precise recommendations.
            </div>
            """, unsafe_allow_html=True)
            
            # Create columns for input groups
            col1, col2 = st.columns(2)
            
            # Store user inputs
            user_inputs = {}
            
            with col1:
                st.subheader("Environmental Conditions")
                
                # Environmental inputs - checking if they exist in the model's features
                env_features = [
                    'Temperature_min', 'Temperature_max', 
                    'Humidity_min', 'Humidity_max',
                    'Annual rainfall (mm)', 'Rainfall',
                    'Moisture_min', 'Moisture_max'
                ]
                
                for feature in env_features:
                    if feature in recommender.relevant_features:
                        user_inputs[feature] = st.number_input(
                            feature, 
                            min_value=0.0,
                            value=20.0 if 'Temperature' in feature else 50.0,
                            step=0.1,
                            key=f"adv_{feature}"
                        )
            
            with col2:
                st.subheader("Soil Characteristics")
                
                # Soil inputs - checking if they exist in the model's features
                soil_features = ['N', 'P', 'K', 'pH_min', 'pH_max']
                
                for feature in soil_features:
                    if feature in recommender.relevant_features:
                        max_val = 14.0 if 'pH' in feature else 200.0
                        user_inputs[feature] = st.number_input(
                            feature, 
                            min_value=0.0, 
                            max_value=max_val,
                            value=7.0 if 'pH' in feature else 50.0,
                            step=0.1,
                            key=f"adv_{feature}"
                        )
            
            # Calendar features
            calendar_features = [f for f in recommender.relevant_features if 'Crop calendar' in f]
            if calendar_features:
                st.subheader("Planting Season")
                
                # Split into start and end months
                start_months = [f for f in calendar_features if 'start' in f]
                end_months = [f for f in calendar_features if 'end' in f]
                
                # Create columns for month selection
                month_cols = st.columns(4)
                
                # Start months
                if start_months:
                    for i, feature in enumerate(start_months):
                        month_name = feature.split('_')[-1] if '_' in feature else feature
                        # Create a truly unique key by combining feature name with index
                        unique_key = f"start_{feature}_{i}"
                        user_inputs[feature] = month_cols[i % 4].checkbox(
                            f"Start: {month_name}",
                            value=False,
                            key=unique_key
                        )
                
                # End months
                if end_months:
                    for i, feature in enumerate(end_months):
                        month_name = feature.split('_')[-1] if '_' in feature else feature
                        # Create a truly unique key by combining feature name with index
                        unique_key = f"end_{feature}_{i}"
                        user_inputs[feature] = month_cols[i % 4].checkbox(
                            f"End: {month_name}",
                            value=False,
                            key=unique_key
                        )
            
            # Get detailed recommendations button
            if st.button("Get Detailed Recommendations", type="primary", key="adv_rec_button"):
                with st.spinner("Analyzing conditions and generating detailed recommendations..."):
                    # Prepare input data
                    input_data = pd.DataFrame({
                        feature: [user_inputs.get(feature, 0)] 
                        for feature in recommender.relevant_features
                    })
                    
                    # Add geographic location if available
                    if 'Longitude' in recommender.relevant_features and 'Latitude' in recommender.relevant_features:
                        input_data['Longitude'] = longitude
                        input_data['Latitude'] = latitude
                    
                    # Get recommendations
                    recommendations = recommender.recommend_crop(input_data)
                    
                    if recommendations:
                        # Display recommendations
                        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                        st.subheader("Recommended Crops Based on Your Conditions")
                        
                        # Create columns for top recommendations
                        top_n = min(3, len(recommendations))
                        rec_cols = st.columns(top_n)
                        
                        for i, (col, rec) in enumerate(zip(rec_cols, recommendations[:top_n])):
                            col.metric(
                                label=f"{i+1}. {rec['crop']}", 
                                value=f"{rec['confidence']:.1%}"
                            )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Display bar chart of recommendations
                        st.subheader("Crop Suitability Analysis")
                        chart_data = pd.DataFrame({
                            'Crop': [rec['crop'] for rec in recommendations],
                            'Confidence': [rec['confidence'] for rec in recommendations]
                        })
                        
                        rec_fig = px.bar(
                            chart_data, 
                            x='Crop', 
                            y='Confidence',
                            color='Confidence',
                            color_continuous_scale='Viridis',
                            title='Model Confidence for Each Crop',
                            text_auto='.1%'
                        )
                        
                        rec_fig.update_layout(
                            xaxis_title='Crop',
                            yaxis_title='Confidence Score',
                            yaxis=dict(range=[0, 1])
                        )
                        
                        st.plotly_chart(rec_fig)
                        
                        # Display technical details for expert mode
                        if mode == "Expert Mode":
                            with st.expander("Technical Details"):
                                st.write("**Model Input Values:**")
                                st.dataframe(input_data)
                                
                                st.write("**Feature Importance for Top Recommendation:**")
                                if recommender.feature_importance_df is not None:
                                    st.dataframe(recommender.feature_importance_df.head(10))
                    else:
                        st.error("Unable to generate recommendations based on these conditions. Please adjust your inputs and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
    <h3>About This System</h3>
    <p>This crop recommendation system uses machine learning to provide personalized crop suggestions 
    based on environmental factors, soil conditions, and geolocation data. The system analyzes 
    patterns in historical crop performance to make intelligent recommendations.</p>
    <p><small>© 2025 Agricultural AI Solutions</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    farmer_friendly_interface()