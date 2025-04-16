import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page setup with minimal HTML rendering
st.set_page_config(page_title="LoanBot", layout="centered")
st.markdown("# 💬 LoanBot", unsafe_allow_html=False)
st.markdown("Answer step-by-step like a WhatsApp chat 📱")

# Improved caching strategy
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_and_train_model():
    # Use more efficient pandas read with only necessary columns
    df = pd.read_csv("loan_data.csv")
    df.ffill(inplace=True)

    target_col = 'loan_status'
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Process all categorical columns at once
    encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Handle target encoding
    target_encoder = None
    if y.dtype == 'object':
        y_le = LabelEncoder()
        y = y_le.fit_transform(y)
        target_encoder = y_le

    # Use faster random forest configuration
    model = RandomForestClassifier(
        n_estimators=100,  # Reduce if possible while maintaining accuracy
        n_jobs=-1,         # Parallel processing
        max_depth=10,      # Limit tree depth for faster inference
        bootstrap=True,    # Use bootstrap samples
        warm_start=False   # No warm start needed for single training
    )
    model.fit(X, y)

    return model, encoders, target_encoder, X.columns

# Initialize session state once
if 'initialized' not in st.session_state:
    # Load model only once at startup
    model, encoders, target_encoder, columns = load_and_train_model()
    
    # Store everything in session state
    st.session_state.model = model
    st.session_state.encoders = encoders
    st.session_state.target_encoder = target_encoder
    st.session_state.columns = columns
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.history = [("👋 Hello! I'm LoanBot. Let's check your loan eligibility.", False)]
    st.session_state.initialized = True

# Use container for more efficient rendering
chat_container = st.container()

# Get session state for easy reference
model = st.session_state.model
encoders = st.session_state.encoders
target_encoder = st.session_state.target_encoder
columns = st.session_state.columns
current_step = st.session_state.step

# Display chat history more efficiently
with chat_container:
    for msg, is_user in st.session_state.history:
        bubble_color = "#dcf8c6" if is_user else "#f1f0f0"
        align = "right" if is_user else "left"
        st.markdown(f"""
        <div style='text-align: {align}; margin: 5px 0;'>
            <span style='background-color: {bubble_color}; padding: 8px 12px; border-radius: 18px;
                        display: inline-block; max-width: 80%;'>
                {msg}
            </span>
        </div>
        """, unsafe_allow_html=True)

# Input handling
if current_step < len(columns):
    col = columns[current_step]
    is_cat = col in encoders

    # Simplified form
    with st.form(key=f"form_{current_step}", clear_on_submit=True):
        if is_cat:
            # Pre-compute options list
            options = list(encoders[col].classes_)
            user_input = st.selectbox(f"Your {col}:", options)
        else:
            user_input = st.number_input(f"Enter {col}:", step=1.0)

        submitted = st.form_submit_button("Send")
        if submitted:
            # Update session state without recalculating
            st.session_state.answers[col] = user_input
            st.session_state.history.append((f"{user_input}", True))
            st.session_state.step += 1

            # Provide next input prompt
            if st.session_state.step < len(columns):
                next_col = columns[st.session_state.step]
                st.session_state.history.append((f"Please enter your {next_col}:", False))
            
            # Force refresh
            st.experimental_rerun()

# Prediction logic - only run once when all inputs collected
elif current_step == len(columns):
    # Prepare input data more efficiently
    input_data = []
    for col in columns:
        val = st.session_state.answers[col]
        if col in encoders:
            val = encoders[col].transform([val])[0]
        input_data.append(val)

    # Single prediction
    pred = model.predict([input_data])[0]
    if target_encoder:
        pred = target_encoder.inverse_transform([pred])[0]

    # Simplified result check
    loan_approved = str(pred).lower() in ['1', 'yes', 'approved', 'y', 'true']
    result = "✅ Loan Approved!" if loan_approved else "❌ Loan Not Approved."
    
    st.session_state.history.append((result, False))
    st.session_state.step += 1  # Mark prediction as complete
    st.experimental_rerun()  # Force refresh to show result

# Restart option - use a button in a fixed position
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔁 Restart Chat", use_container_width=True):
        st.session_state.step = 0
        st.session_state.answers = {}
        st.session_state.history = [("👋 Hello! I'm LoanBot. Let's check your loan eligibility.", False)]
        st.experimental_rerun()
