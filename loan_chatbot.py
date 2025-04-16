import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Minimal page setup - absolute minimum UI elements
st.set_page_config(page_title="LoanBot", layout="wide")
st.markdown("# LoanBot")

# Model loading with file caching instead of Streamlit caching
MODEL_PATH = "loan_model.pkl"
ENCODERS_PATH = "encoders.pkl"

def fast_train_model():
    """Train model and save to disk for fastest loading"""
    df = pd.read_csv("loan_data.csv", engine='c')
    
    target_col = 'loan_status'
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Process categoricals
    encoders = {}
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = pd.factorize(X[col])[0]
        encoders[col] = dict(zip(df[col].unique(), pd.factorize(df[col])[0]))

    # Faster than sklearn for binary classification
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(n_jobs=-1, force_row_wise=True)
    model.fit(X, y)
    
    # Save model and encoders to disk
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(ENCODERS_PATH, 'wb') as f:
        pickle.dump((encoders, X.columns), f)

def load_model():
    """Fast load from disk cache"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
        fast_train_model()
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, 'rb') as f:
        encoders, columns = pickle.load(f)
    
    return model, encoders, columns

# Initialize only once - ultra minimal state
if 'step' not in st.session_state:
    # Load model on first run
    model, encoders, columns = load_model()
    
    # Store minimal references
    st.session_state.model = model
    st.session_state.encoders = encoders
    st.session_state.columns = columns
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.messages = []
    
    # Add first message
    st.session_state.messages.append({"content": "Hello! I'm LoanBot. Let's check your loan eligibility.", "role": "assistant"})

# Fast direct access
model = st.session_state.model
encoders = st.session_state.encoders
columns = st.session_state.columns
current_step = st.session_state.step

# Display chat history with native Streamlit elements (faster)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Simple input handling - just the essentials
if current_step < len(columns):
    col = columns[current_step]
    is_cat = col in encoders
    
    # Pre-computation
    prompt = f"Enter your {col}:"
    
    # Fast input method with no form
    if is_cat:
        options = list(encoders[col].keys())
        user_input = st.selectbox(prompt, options, key=f"input_{current_step}")
        submit_button = st.button("Send", key=f"submit_{current_step}")
    else:
        user_input = st.number_input(prompt, step=1.0, key=f"input_{current_step}")
        submit_button = st.button("Send", key=f"submit_{current_step}")
    
    if submit_button:
        # Store answer
        st.session_state.answers[col] = user_input
        st.session_state.messages.append({"content": f"{user_input}", "role": "user"})
        st.session_state.step += 1
        
        # Prompt for next step
        if st.session_state.step < len(columns):
            next_col = columns[st.session_state.step]
            st.session_state.messages.append({"content": f"Please enter your {next_col}:", "role": "assistant"})
        
        st.experimental_rerun()

# Fast prediction logic - minimal processing
elif current_step == len(columns):
    # Prepare input as numpy array for speed
    input_data = np.zeros(len(columns))
    
    for i, col in enumerate(columns):
        val = st.session_state.answers[col]
        if col in encoders:
            input_data[i] = encoders[col].get(val, 0)
        else:
            input_data[i] = float(val)
    
    # Single fast prediction
    input_array = input_data.reshape(1, -1)
    pred = model.predict(input_array)[0]
    
    result = "✅ Loan Approved!" if int(pred) == 1 else "❌ Loan Not Approved."
    st.session_state.messages.append({"content": result, "role": "assistant"})
    st.session_state.step += 1
    st.experimental_rerun()

# Streamlined restart - single button
if st.button("Restart", key="restart_button"):
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.messages = [{"content": "Hello! I'm LoanBot. Let's check your loan eligibility.", "role": "assistant"}]
    st.experimental_rerun()
