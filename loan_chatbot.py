import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load and process data
df = pd.read_csv("loan_data.csv")

# Fill missing values
df.ffill(inplace=True)

# Identify target column
target_col = 'loan_status'  # <-- change if different
X = df.drop(target_col, axis=1)
y = df[target_col]

# Encode categorical columns
encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Encode target if needed
if y.dtype == 'object':
    y_le = LabelEncoder()
    y = y_le.fit_transform(y)
    target_encoder = y_le
else:
    target_encoder = None

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit app
def loan_chatbot():
    st.title("Loan Approval Prediction Bot")

    st.write("Please provide the following details for the loan application:")

    user_input = []

    # Collect user input
    for col in X.columns:
        if col in encoders:
            options = list(encoders[col].classes_)
            val = st.selectbox(f"{col} (e.g., {', '.join(options)})", options)
            encoded_val = encoders[col].transform([val])[0]
            user_input.append(encoded_val)
        else:
            val = st.number_input(f"{col} (numeric):", format="%.2f")
            user_input.append(val)

    # Submit button to process the input and make predictions
    if st.button("Predict Loan Approval"):
        # Make prediction
        prediction = model.predict([user_input])[0]

        # Decode prediction
        if target_encoder:
            prediction = target_encoder.inverse_transform([prediction])[0]

        # Show result
        if str(prediction).lower() in ['1', 'yes', 'approved', 'y']:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Not Approved")

if __name__ == "__main__":
    loan_chatbot()
