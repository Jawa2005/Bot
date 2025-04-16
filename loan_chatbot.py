import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(page_title="LoanBot", layout="centered")
st.markdown("<h1 style='text-align: center;'>💬 LoanBot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Answer step-by-step like a WhatsApp chat 📱</p>", unsafe_allow_html=True)

@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("loan_data.csv")
    df.ffill(inplace=True)

    # Show available columns to help debugging if needed
    st.write("Available columns in data:", df.columns.tolist())

    # Try common options, and select only those present in the CSV
    possible_columns = ['LoanAmount', 'ApplicantIncome', 'EmploymentStatus', 'CreditScore', 'loan_amount', 'income', 'employment_status', 'credit_score']
    selected_columns = [col for col in possible_columns if col in df.columns]

    if not selected_columns:
        st.error("❌ None of the selected columns are found in the dataset. Please check column names.")
        st.stop()

    target_col = 'loan_status'
    if target_col not in df.columns:
        st.error(f"❌ '{target_col}' column not found in dataset.")
        st.stop()

    X = df[selected_columns]
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

    model = RandomForestClassifier(n_jobs=-1)
    model.fit(X, y)

    return model, encoders, target_encoder, selected_columns

# Initial session state
if 'model' not in st.session_state:
    model, encoders, target_encoder, columns = load_and_train_model()
    st.session_state.model = model
    st.session_state.encoders = encoders
    st.session_state.target_encoder = target_encoder
    st.session_state.columns = columns
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.predicted = False
    st.session_state.history = [("👋 Hello! I’m LoanBot. Let’s check your loan eligibility.", False)]

# Access session state
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

# Step-by-step input collection
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
            st.session_state.answers[col] = user_input
            st.session_state.history.append((f"{user_input}", True))
            st.session_state.step += 1

            if st.session_state.step < len(columns):
                next_col = columns[st.session_state.step]
                st.session_state.history.append((f"Please enter your {next_col}:", False))

# Prediction
elif not st.session_state.predicted:
    input_data = []
    for col in columns:
        val = st.session_state.answers[col]
        if col in encoders:
            val = encoders[col].transform([val])[0]
        input_data.append(val)

    pred = model.predict([input_data])[0]
    if target_encoder:
        pred = target_encoder.inverse_transform([pred])[0]

    result = "✅ Loan Approved!" if str(pred).lower() in ['1', 'yes', 'approved', 'y'] else "❌ Loan Not Approved."
    st.session_state.history.append((result, False))
    st.session_state.predicted = True

# Restart button
if st.button("🔁 Restart Chat"):
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.predicted = False
    st.session_state.history = [("👋 Hello! I’m LoanBot. Let’s check your loan eligibility.", False)]
