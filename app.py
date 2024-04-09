import streamlit as st
from catboost import CatBoostClassifier
from sklearn.preprocessing import RobustScaler
import numpy as np

# Load the CatBoost model (replace 'catboost_model.bin' with your actual file path)
model = CatBoostClassifier()
model.load_model('catboost_model.bin')  # Adjust the file name and path as necessary

# Feature name mapping for clarity in the app
feature_names = {
    'Sex_x': 'Gender',
    'mar_x': 'Marital_status',
    'Odesc_x': 'Occupation',
    'Cust_class_x': 'Customer_category',
    'postcode': 'Postal_code',
    'age': 'Age'
}

st.title("Bad Debt Prediction App")
st.write("This app uses a CatBoost model to predict the likelihood of bad debt based on customer information.")

# Create a RobustScaler object
scaler = RobustScaler()

# Input fields for each feature
Gender = st.selectbox(feature_names['Sex_x'], ['Male', 'Female'])
Marital_status = st.selectbox(feature_names['mar_x'], ['Married', 'Single', 'Divorced', 'Widowed',
                                                         'Separated', 'Cohabiting', 'Commonlaw',
                                                         'Estranged', 'Patnered', 'CivilPart'])
Occupation = st.text_input(feature_names['Odesc_x'])
Customer_category = st.selectbox(feature_names['Cust_class_x'], ['Mixed', 'Commercial', 'Consumer', 'Micro'])
Postal_code = st.text_input(feature_names['postcode'])
Age = st.number_input(feature_names['age'])

# Button to trigger prediction
if st.button("Predict Bad Debt Risk"):

    # Handle missing categorical features (replace with your logic for identifying missing values)
    missing_features = [key for key, value in locals().items() if key in feature_names.values() and (value is None or value == '')]

    # Add placeholder values for missing features (if applicable)
    if missing_features:
        for feature in missing_features:
            locals()[feature] = "Unknown"  # You can define a more suitable placeholder

    # Prepare data for prediction
    scaled_age = scaler.fit_transform(np.array([Age]).reshape(-1, 1))
    data = [Gender, Marital_status, Occupation, Customer_category, Postal_code, scaled_age[0][0]]

    # Make prediction
    pred = model.predict(data)

    if pred == 'good debt':
        st.success("There is a low risk of bad debt for this customer.")
    else:
        st.error("There is a high risk of bad debt for this customer.")
