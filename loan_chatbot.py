import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time

# Page setup
st.set_page_config(page_title="LoanBot", layout="centered")
st.markdown("<h1 style='text-align: center;'>💬 LoanBot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Answer step-by-step like a WhatsApp chat 📱</p>", unsafe_allow_html=True)

# Cache dataset loading and model training to avoid re-loading on each interaction
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("loan_data.csv")
    df.ffill(inplace=True)

    target_col = 'loan_status'
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    if y.dtype == 'object':
        y_le = LabelEncoder()
        y = y_le.fit_transform(y)
        target_encoder = y_le
    else:
        target_encoder = None

    model = RandomForestClassifier(n_jobs=-1)  # Enable parallel processing in RandomForest
    model.fit(X, y)

    return model, encoders, target_encoder, X.columns

# Load model and encoders, store them in session state
if 'model' not in st.session_state:
    model, encoders, target_encoder, columns = load_and_train_model()
    st.session_state.model = model
    st.session_state.encoders = encoders
    st.session_state.target_encoder = target_encoder
    st.session_state.columns = columns
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.history = [("👋 Hello! I’m LoanBot. Let’s check your loan eligibility.", False)]

# Get session state for easy reference
model = st.session_state.model
encoders = st.session_state.encoders
target_encoder = st.session_state.target_encoder
columns = st.session_state.columns
current_step = st.session_state.step

# Display chat history
for msg, is_user in st.session_state.history:
    bubble_color = "#dcf8c6" if is_user else "#f1f0f0"
    align = "right" if is_user else "left"
    st.markdown(f"""
    <div style='text-align: {align}; margin: 10px 0;'>
        <span style='background-color: {bubble_color}; padding: 10px 15px; border-radius: 20px;
                     display: inline-block; max-width: 80%;'>
            {msg}
        </span>
    </div>
    """, unsafe_allow_html=True)

# Input step-by-step without delays
if current_step < len(columns):
    col = columns[current_step]
    is_cat = col in encoders

    with st.form(key=f"form_{current_step}", clear_on_submit=True):
        if is_cat:
            options = list(encoders[col].classes_)
            user_input = st.selectbox(f"Your {col}:", options)
        else:
            user_input = st.number_input(f"Enter {col}:", step=1.0)

        submitted = st.form_submit_button("Send")
        if submitted:
            # Collect user input and update session state
            st.session_state.answers[col] = user_input
            st.session_state.history.append((f"{user_input}", True))
            st.session_state.step += 1

            # Provide next input prompt
            if st.session_state.step < len(columns):
                next_col = columns[st.session_state.step]
                st.session_state.history.append((f"Please enter your {next_col}:", False))

    # Avoid any unnecessary re-runs or delays at this stage
else:
    # All inputs collected, make prediction
    input_data = []
    for col in columns:
        val = st.session_state.answers[col]
        if col in encoders:
            val = encoders[col].transform([val])[0]
        input_data.append(val)

    # Run prediction immediately after inputs are collected
    pred = model.predict([input_data])[0]
    if target_encoder:
        pred = target_encoder.inverse_transform([pred])[0]

    result = "✅ Loan Approved!" if str(pred).lower() in ['1', 'yes', 'approved', 'y'] else "❌ Loan Not Approved."
    st.session_state.history.append((result, False))
    st.session_state.step += 1  # Prevent re-running prediction

# Restart option
if st.button("🔁 Restart Chat"):
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.history = [("👋 Hello! I’m LoanBot. Let’s check your loan eligibility.", False)]
