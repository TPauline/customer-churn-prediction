# app.py — Customer Churn Prediction Dashboard
# Run with: streamlit run app.py   (from the churn-prediction/ root folder)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — ** first Streamlit command in the file**
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL AND SUPPORT FILES
# @st.cache_resource → loads once and reuses on every user interaction
# Without caching the model would reload on every button click (very slow)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('models/churn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    with open('models/feature_columns.json', 'r') as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    return df

model, scaler, feature_cols = load_model()
df_raw = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Apply the same preprocessing used during training
# Must mirror exactly what was done in 02_feature_engineering.ipynb
# ─────────────────────────────────────────────────────────────────────────────
def prepare_features(df_input):
    df = df_input.copy()

    binary_map  = {'Yes': 1, 'No': 0}
    service_map = {'Yes': 1, 'No': 0,
                   'No phone service': 0, 'No internet service': 0}

    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0)

    for col in ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        if col in df.columns:
            df[col] = df[col].map(service_map).fillna(0)

    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0}).fillna(0)

    df = pd.get_dummies(df,
                        columns=['Contract', 'PaymentMethod', 'InternetService'],
                        drop_first=True)

    # Add any missing dummy columns as 0
    # Must match training column set exactly
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    mode = st.selectbox(
        "View Mode",
        ["📋 Dataset Overview", "👤 Predict Single Customer"]
    )
    threshold = st.slider(
        "Risk Threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Customers above this probability are flagged as high risk"
    )
    st.caption(f"Flagging customers with >{int(threshold * 100)}% churn probability")
    st.divider()
    st.markdown("**About**")
    st.markdown("Logistic Regression · Telco Churn dataset · SMOTE balanced")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Identify customers at risk of cancelling before they leave.")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# VIEW 1: DATASET OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if mode == "📋 Dataset Overview":

    df_features  = df_raw.drop(columns=['customerID', 'Churn'])
    X_all        = prepare_features(df_features)
    X_all_scaled = scaler.transform(X_all)
    proba_all    = model.predict_proba(X_all_scaled)[:, 1]

    total          = len(df_raw)
    churned_actual = (df_raw['Churn'] == 'Yes').sum()
    high_risk      = (proba_all >= threshold).sum()
    avg_risk       = proba_all.mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers",   f"{total:,}")
    c2.metric("Actual Churn Rate", f"{churned_actual / total * 100:.1f}%")
    c3.metric("High Risk Flagged", f"{high_risk:,}",
              delta=f"{high_risk / total * 100:.1f}% of base",
              delta_color="inverse")
    c4.metric("Avg Risk Score",    f"{avg_risk:.1f}%")

    st.divider()
    st.subheader("Highest Risk Customers")

    df_results = df_raw.copy()
    df_results['Churn Probability (%)'] = (proba_all * 100).round(1)
    df_results['Risk Level'] = np.where(
        proba_all >= threshold, 'High',
        np.where(proba_all >= 0.35, 'Medium', 'Low')
    )

    df_high      = df_results[proba_all >= threshold].sort_values(
        'Churn Probability (%)', ascending=False
    )
    display_cols = ['customerID', 'Contract', 'tenure',
                    'MonthlyCharges', 'Churn Probability (%)', 'Risk Level']
    st.dataframe(df_high[display_cols].head(50),
                 use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Risk Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(proba_all, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f"Threshold ({int(threshold * 100)}%)")
        ax.set_xlabel('Predicted Churn Probability')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Churn Risk Score Distribution')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        contract_data = df_raw.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        contract_data.columns = ['Contract', 'Churn Rate (%)']

        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(contract_data['Contract'], contract_data['Churn Rate (%)'],
                      color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='white')
        for bar, val in zip(bars, contract_data['Churn Rate (%)']):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%", ha='center', fontweight='bold')
        ax.set_title('Historical Churn Rate by Contract Type')
        ax.set_ylabel('Churn Rate (%)')
        ax.set_ylim(0, 55)
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# VIEW 2: SINGLE CUSTOMER PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "👤 Predict Single Customer":

    st.subheader("Enter Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Account**")
        gender     = st.selectbox("Gender", ["Male", "Female"])
        senior     = st.selectbox("Senior Citizen", [0, 1],
                                   format_func=lambda x: "Yes" if x else "No")
        partner    = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure     = st.slider("Tenure (months)", 0, 72, 12)
        contract   = st.selectbox("Contract",
                                   ["Month-to-month", "One year", "Two year"])
        payment    = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless  = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col2:
        st.markdown("**Services**")
        phone    = st.selectbox("Phone Service", ["Yes", "No"])
        multi    = st.selectbox("Multiple Lines",
                                 ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security",
                                 ["Yes", "No", "No internet service"])
        backup   = st.selectbox("Online Backup",
                                 ["Yes", "No", "No internet service"])
        device   = st.selectbox("Device Protection",
                                 ["Yes", "No", "No internet service"])
        tech     = st.selectbox("Tech Support",
                                 ["Yes", "No", "No internet service"])
        tv       = st.selectbox("Streaming TV",
                                 ["Yes", "No", "No internet service"])
        movies   = st.selectbox("Streaming Movies",
                                 ["Yes", "No", "No internet service"])
        monthly  = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
        total_ch = monthly * tenure
        st.caption(f"Estimated Total Charges: ${total_ch:,.2f}")

    if st.button("Predict Churn Risk", type="primary"):

        customer_data = pd.DataFrame([{
            'gender': gender,         'SeniorCitizen': senior,
            'Partner': partner,       'Dependents': dependents,
            'tenure': tenure,         'PhoneService': phone,
            'MultipleLines': multi,   'InternetService': internet,
            'OnlineSecurity': security, 'OnlineBackup': backup,
            'DeviceProtection': device, 'TechSupport': tech,
            'StreamingTV': tv,        'StreamingMovies': movies,
            'Contract': contract,     'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly, 'TotalCharges': total_ch
        }])

        X_cust        = prepare_features(customer_data)
        X_cust_scaled = scaler.transform(X_cust)
        churn_prob    = model.predict_proba(X_cust_scaled)[0, 1]
        churn_pct     = churn_prob * 100

        st.divider()
        r1, r2 = st.columns([1, 2])

        with r1:
            if churn_prob >= threshold:
                st.error(f"HIGH RISK — {churn_pct:.1f}% churn probability")
                st.write("Consider a retention offer immediately.")
            elif churn_prob >= 0.35:
                st.warning(f"MEDIUM RISK — {churn_pct:.1f}% churn probability")
                st.write("Monitor this customer closely.")
            else:
                st.success(f"LOW RISK — {churn_pct:.1f}% churn probability")
                st.write("This customer appears stable.")

        with r2:
            bar_color = ('#e74c3c' if churn_prob >= threshold
                         else '#f39c12' if churn_prob >= 0.35
                         else '#2ecc71')
            fig, ax = plt.subplots(figsize=(5, 1.8))
            ax.barh(['Risk'], [churn_pct], color=bar_color, height=0.4)
            ax.barh(['Risk'], [100 - churn_pct], left=[churn_pct],
                    color='#ecf0f1', height=0.4)
            ax.axvline(threshold * 100, color='black', linestyle='--',
                       linewidth=1.5,
                       label=f"Threshold ({int(threshold * 100)}%)")
            ax.set_xlim(0, 100)
            ax.set_xlabel('Churn Probability (%)')
            ax.set_title('Risk Score Gauge')
            ax.legend()
            st.pyplot(fig)
            plt.close()