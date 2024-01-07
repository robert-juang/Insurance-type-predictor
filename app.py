import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from fuzzywuzzy import process

# Load the Random Forest model
rf_model = joblib.load("random_forest_model.joblib")

# Load the LabelEncoders for other categorical columns
encoders = {}
categorical_columns = ['Company', 'Coverage', 'SubCoverage', 'SubReason', 'Disposition', 'Conclusion', 'Status']
for col in categorical_columns:
    encoder_filename = f"encoder files/{col}_encoder.joblib"
    encoders[col] = joblib.load(encoder_filename)

def fuzzy_match(value, encoder):
    # Get the closest match using fuzzywuzzy
    matches = process.extractOne(value, encoder.classes_)
    closest_match, score = matches[0], matches[1]
    
    # Check if the score is above a certain threshold (adjust as needed)
    if score >= 80:
        return closest_match
    else:
        st.warning(f"No close match found for {value}. Please provide a valid value.")
        return None

def encode_values(input_values):
    encoded_values = {}
    for col, value in input_values.items():
        # Check if the input value is not empty
        if value:
            if col == 'Recovery':
                # Handle 'Recovery' separately
                encoded_values[col] = float(value) if value else None
            else:
                # Use fuzzy matching for other categorical columns
                encoder = encoders[col]
                closest_match = fuzzy_match(value, encoder)
                
                if closest_match is not None:
                    encoded_values[col] = encoder.transform([closest_match])[0]
                else:
                    return None
        else:
            st.warning(f"Input value for {col} is empty. Please provide a valid value.")
            return None

    return encoded_values

def predict_reason(encoded_values):
    input_df = pd.DataFrame([encoded_values])
    # Rearrange columns to match the order during training
    input_df = input_df[['Company', 'Coverage', 'SubCoverage', 'SubReason', 'Disposition', 'Conclusion', 'Recovery', 'Status']]
    prediction = rf_model.predict(input_df)[0]
    probabilities = rf_model.predict_proba(input_df)[0]
    return prediction, max(probabilities)

# Streamlit app
st.title("Insurance Reason Predictor")

# Welcome message
st.write("Welcome to the Insurance Reason Predictor app! Enter the values below to predict the most probable reason.")

# User input form
user_input = {}
for col in categorical_columns + ['Recovery']:
    user_input[col] = st.text_input(f"Enter {col}:", "")

# Encode values with fuzzy matching
encoded_input = encode_values(user_input)

# Predict reason
if st.button("Predict Reason"):
    # Ensure input values are not empty before proceeding
    if any(encoded_input.values()):
        prediction, probability = predict_reason(encoded_input)
        st.success(f"Predicted Reason: {prediction}")
        st.write(f"Probability: {probability:.2%} (Most Probable Reason)")
    else:
        st.warning("Please provide values for the input fields before predicting.")
