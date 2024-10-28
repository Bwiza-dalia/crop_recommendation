#!/usr/bin/env python
# coding: utf-8

# # Import required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# # Load and examine the data

# In[2]:


df = pd.read_csv('sample_dataset2.xlsx - Sheet1.csv')


# # Data Preprocessing

# In[5]:


# Function to clean numeric ranges
def clean_numeric_range(value):
    if isinstance(value, str):
        # If the value contains a range (e.g., '156-350'), take the average
        if '-' in value:
            low, high = map(float, value.split('-'))
            return (low + high) / 2
    return value


# In[6]:


# Data cleaning function
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


# In[7]:


# Clean the dataset
df_cleaned = clean_dataset(df)


# In[17]:


print("Cleaned Dataset Info:")
print("-" * 50)
print(df_cleaned.info())


# In[ ]:





# # Create the CropRecommendationSystem class

# In[9]:


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
            , 'Annual rainfall (mm)', 'Soil type',
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


# In[10]:


def analyze_and_visualize(recommender, X, y, y_pred):
    """Analyze and visualize model performance"""
    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2%}")
    
    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, 
                              target_names=recommender.label_encoder.classes_))
    
    # Plot feature importance
    plt.figure(figsize=(15, 8))
    feature_importance = recommender.get_feature_importance()
    top_features = feature_importance.head(20)  # Show top 20 features
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title('Top 20 Most Important Features in Crop Recommendation')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    # Plot confusion matrix for top crops
    top_crops = pd.Series(recommender.label_encoder.inverse_transform(y)).value_counts().head(10).index
    mask = pd.Series(recommender.label_encoder.inverse_transform(y)).isin(top_crops)
    y_filtered = y[mask]
    y_pred_filtered = y_pred[mask]
    
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_filtered, y_pred_filtered)
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=top_crops,
                yticklabels=top_crops)
    plt.title('Confusion Matrix (Top 10 Crops)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[11]:


# Initialize and train the model
recommender = CropRecommendationSystem()


# In[12]:


X, y = recommender.preprocess_data(df_cleaned)


# In[13]:


X_train, X_test, y_train, y_test, y_pred = recommender.train_model(X, y)


# In[14]:


# Analyze and visualize results
analyze_and_visualize(recommender, y_test, y_test, y_pred)


# In[15]:


# Function to get recommendations
def get_recommendations(conditions):
    recommendations = recommender.recommend_crop(conditions)
    print("\nTop 3 Recommended Crops:")
    print("-" * 50)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['crop']} (Confidence: {rec['confidence']:.2%})")


# In[16]:


# Prepare a sample from the dataset by retaining only relevant features
sample_data = df_cleaned.iloc[0:1][recommender.relevant_features].copy()

# Get crop recommendations using the sample data
get_recommendations(sample_data)


# In[ ]:




