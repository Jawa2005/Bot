import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("loan_data.csv")
df.ffill(inplace=True)

# Setup
target_col = 'loan_status'
X = df.drop(target_col, axis=1)
y = df[target_col]

# Encode categorical features
encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Encode target
if y.dtype == 'object':
    y_le = LabelEncoder()
    y = y_le.fit_transform(y)
    target_encoder = y_le
else:
    target_encoder = None

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.title("🏦 Loan Approval Prediction App")
st.write("Fill the form below to check if your loan is likely to be approved.")

# User input
user_input = []
for col in X.columns:
    if col in encoders:
        options = list(encoders[col].classes_)
        val = st.selectbox(f"{col.capitalize()}", options)
        encoded_val = encoders[col].transform([val])[0]
        user_input.append(encoded_val)
    else:
        val = st.number_input(f"{col.capitalize()}", value=0.0, step=1.0)
        user_input.append(val)

# Predict button
if st.button("Check Loan Approval"):
    prediction = model.predict([user_input])[0]
    if target_encoder:
        prediction = target_encoder.inverse_transform([prediction])[0]
    if str(prediction).lower() in ['1', 'yes', 'approved', 'y']:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
