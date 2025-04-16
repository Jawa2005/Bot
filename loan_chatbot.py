import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Streamlit page setup
st.set_page_config(page_title="LoanBot", layout="centered")
st.markdown("<h1 style='text-align: center;'>💬 LoanBot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Answer step-by-step like a WhatsApp chat 📱</p>", unsafe_allow_html=True)

@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("loan_data.csv")
    df.ffill(inplace=True)

    # Auto-select relevant columns if present
    target_col = "loan_status"
    possible_features = ['LoanAmount', 'ApplicantIncome', 'EmploymentStatus', 'CreditScore']
    selected_columns = [col for col in possible_features if col in df.columns]

    if target_col not in df.columns or len(selected_columns) < 2:
        st.error("Dataset must contain at least 2 of these columns: " + str(possible_features) + " and 'loan_status'.")
        st.stop()

    X = df[selected_columns]
    y = df[target_col]

    # Encode categorical variables
    encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Encode target if needed
    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
    else:
        target_encoder = None

    # Train model
    model = RandomForestClassifier(n_jobs=-1)
    model.fit(X, y)

    return model, encoders, target_encoder, selected_columns

# Load model once
if "model" not in st.session_state:
    model, encoders, target_encoder, feature_cols = load_and_train_model()
    st.session_state.model = model
    st.session_state.encoders = encoders
    st.session_state.target_encoder = target_encoder
    st.session_state.feature_cols = feature_cols
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.history = [("👋 Hello! I’m LoanBot. Let’s check your loan eligibility.", False)]
    st.session_state.predicted = False

# Load session variables
model = st.session_state.model
encoders = st.session_state.encoders
target_encoder = st.session_state.target_encoder
columns = st.session_state.feature_cols
step = st.session_state.step

# Display chat
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

# Collect input step-by-step
if step < len(columns):
    col = columns[step]
    is_categorical = col in encoders

    with st.form(key=f"form_{step}", clear_on_submit=True):
        if is_categorical:
            options = list(encoders[col].classes_)
            user_input = st.selectbox(f"Your {col}:", options)
        else:
            user_input = st.number_input(f"Enter {col}:", step=1.0)

        submitted = st.form_submit_button("Send")
        if submitted:
            st.session_state.answers[col] = user_input
            st.session_state.history.append((f"{user_input}", True))
            st.session_state.step += 1

            if st.session_state.step < len(columns):
                next_col = columns[st.session_state.step]
                st.session_state.history.append((f"Please enter your {next_col}:", False))

# Predict result once all inputs are collected
elif not st.session_state.predicted:
    input_data = []
    for col in columns:
        val = st.session_state.answers[col]
        if col in encoders:
            val = encoders[col].transform([val])[0]
        input_data.append(val)

    prediction = model.predict([input_data])[0]
    if target_encoder:
        prediction = target_encoder.inverse_transform([prediction])[0]

    result_text = "✅ Loan Approved!" if str(prediction).lower() in ['1', 'yes', 'approved'] else "❌ Loan Not Approved."
    st.session_state.history.append((result_text, False))
    st.session_state.predicted = True

# Restart button
if st.button("🔁 Restart Chat"):
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.history = [("👋 Hello! I’m LoanBot. Let’s check your loan eligibility.", False)]
    st.session_state.predicted = False
