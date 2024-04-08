
import streamlit as st
from catboost import CatBoostClassifier
from sklearn.preprocessing import RobustScaler

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
    data = {
        feature_names['Sex_x']: Gender,
        feature_names['mar_x']: Marital_status,
        feature_names['Odesc_x']: Occupation,
        feature_names['Cust_class_x']: Customer_category,
        feature_names['postcode']: Postal_code,
        feature_names['age']: Age
    }

    # **Handle missing categorical features:**
    # 1. Identify potential missing features (replace with your logic)
    missing_features = [key for key, value in data.items() if value is None or value == '']

    # 2. Add placeholder values for missing features (if applicable)
    if missing_features:
        for feature in missing_features:
            data[feature] = "Unknown"  # You can define a more suitable placeholder

    # Convert dictionary values to a list of feature values
    pred = model.predict(list(data.values()))

    if pred == 'good debt':
        st.success("There is a low risk of bad debt for this customer.")
    else:
        st.error("There is a high risk of bad debt for this customer.")

