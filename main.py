# main.py

# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


try:
    # Attempt to load existing files
    scaler = joblib.load('scaler.pkl')
    rf_classifier = joblib.load('crop_recommendation_model.pkl')
    le = joblib.load('label_encoder.pkl')
    st.success("Model, scaler, and label encoder loaded successfully!")
except FileNotFoundError:
    # If files are not found, initialize and save them
    st.warning("Model, scaler, or label encoder not found. Please train and save them first.")

    # Example data for initializing (use your actual data and training process)
    # This is a placeholder and should be replaced with your actual training code.
    df = pd.read_csv('crop_recommendation.csv')
    df.rename(columns={'label': 'crop'}, inplace=True)

    # Initialize LabelEncoder
    le = LabelEncoder()
    df['crop_encoded'] = le.fit_transform(df['crop'])

    # Prepare features and target
    X = df.drop(['crop', 'crop_encoded'], axis=1)
    y = df['crop_encoded']

    # Initialize and fit the scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize and train the Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_scaled, y)

    # Save the trained model, scaler, and label encoder
    joblib.dump(rf_classifier, 'crop_recommendation_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    
    print("Model, scaler, and label encoder saved successfully!")

# Function to predict crop
def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    """
    Predicts the crop type based on input features.

    Parameters:
    - n (float): Nitrogen content
    - p (float): Phosphorus content
    - k (float): Potassium content
    - temperature (float): Temperature in Celsius
    - humidity (float): Humidity percentage
    - ph (float): Soil pH
    - rainfall (float): Rainfall in mm

    Returns:
    - str: Predicted crop name
    """
    # Create a dataframe from input
    input_data = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = rf_classifier.predict(input_scaled)
    
    # Decode the prediction
    crop = le.inverse_transform(prediction)
    
    return crop[0]

# Streamlit app
def main():
 
    st.title("Crop Recommendation System")
    st.header("Enter the following parameters:")
    n = st.number_input("Nitrogen content (N)", min_value=0.0, max_value=300.0, step=1.0)
    p = st.number_input("Phosphorus content (P)", min_value=0.0, max_value=300.0, step=1.0)
    k = st.number_input("Potassium content (K)", min_value=0.0, max_value=300.0, step=1.0)
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=1.0)

 
    if st.button("Predict Crop"):
        predicted_crop = predict_crop(n, p, k, temperature, humidity, ph, rainfall)
        st.success(f"Predicted Crop: {predicted_crop}")

if __name__ == "__main__":
    main()
