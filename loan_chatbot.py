import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from streamlit_chat import message

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

# Initialize chatbot session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Columns list
columns = list(X.columns)
current_col = columns[st.session_state.step] if st.session_state.step < len(columns) else None

# Title
st.title("💬 LoanBot – WhatsApp Style Loan Predictor")

# Chat history
for chat in st.session_state.chat_history:
    message(chat['msg'], is_user=chat['is_user'])

# Ask next question
if current_col:
    if current_col in encoders:
        options = list(encoders[current_col].classes_)
        user_input = st.selectbox(f"Your answer for '{current_col}':", options, key=current_col)
    else:
        user_input = st.number_input(f"Enter {current_col}:", step=1.0, key=current_col)

    if st.button("Send"):
        # Save user message
        st.session_state.chat_history.append({'msg': str(user_input), 'is_user': True})
        st.session_state.answers[current_col] = user_input
        st.session_state.step += 1

        # Add bot response (next question or result)
        if st.session_state.step < len(columns):
            next_col = columns[st.session_state.step]
            st.session_state.chat_history.append({'msg': f"Please enter your {next_col}:", 'is_user': False})
        else:
            # Predict
            input_values = []
            for col in columns:
                val = st.session_state.answers[col]
                if col in encoders:
                    val = encoders[col].transform([val])[0]
                input_values.append(val)
            pred = model.predict([input_values])[0]
            if target_encoder:
                pred = target_encoder.inverse_transform([pred])[0]
            if str(pred).lower() in ['1', 'yes', 'approved', 'y']:
                st.session_state.chat_history.append({'msg': "✅ Loan Approved", 'is_user': False})
            else:
                st.session_state.chat_history.append({'msg': "❌ Loan Not Approved", 'is_user': False})
