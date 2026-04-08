import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Telecom Churn Intelligence Dashboard",
    page_icon="📉",
    layout="wide"
)

# =========================================================
# CUSTOM CSS FOR PREMIUM UI
# =========================================================
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    .custom-card {
        background-color: #161B22;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
        margin-bottom: 15px;
    }

    .section-header {
        font-size: 28px;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 10px;
    }

    .sub-header {
        font-size: 18px;
        font-weight: 600;
        color: #CBD5E1;
        margin-bottom: 10px;
    }

    .highlight-text {
        color: #38BDF8;
        font-weight: 600;
    }

    .small-note {
        font-size: 14px;
        color: #94A3B8;
    }

    hr {
        border: 1px solid #1E293B;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def risk_badge(risk_level):
    if risk_level == "Very High Risk":
        return "🔴 Very High Risk"
    elif risk_level == "High Risk":
        return "🟠 High Risk"
    elif risk_level == "Medium Risk":
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"


def generate_business_summary(row):
    churn_prob = row["churn_probability"] * 100
    segment = row["customer_segment"]
    risk = row["risk_level"]
    action = row["retention_action"]
    strategy = row["segment_strategy"]

    summary = f"""
    This customer belongs to the **{segment}** segment and currently shows a **{risk.lower()}**
    with an estimated churn probability of **{churn_prob:.2f}%**.

    **Recommended Retention Action:** {action}

    **Segment-Level Business Strategy:** {strategy}
    """
    return summary


@st.cache_resource
def load_prediction_functions():
    from src.predict import predict_single_customer, predict_customer
    return predict_single_customer, predict_customer


@st.cache_data
def get_sample_csv():
    sample_data = pd.DataFrame([
        {
            "account_length": 120,
            "international_plan": "no",
            "voice_mail_plan": "yes",
            "voice_mail_messages": 25,
            "total_day_minutes": 250.5,
            "total_day_calls": 110,
            "total_day_charge": 42.59,
            "total_eve_minutes": 180.2,
            "total_eve_calls": 95,
            "total_eve_charge": 15.32,
            "total_night_minutes": 220.4,
            "total_night_calls": 100,
            "total_night_charge": 9.92,
            "total_intl_minutes": 12.5,
            "total_intl_calls": 4,
            "total_intl_charge": 3.38,
            "number_customer_service_calls": 2
        },
        {
            "account_length": 45,
            "international_plan": "yes",
            "voice_mail_plan": "no",
            "voice_mail_messages": 0,
            "total_day_minutes": 320.0,
            "total_day_calls": 130,
            "total_day_charge": 54.40,
            "total_eve_minutes": 260.0,
            "total_eve_calls": 110,
            "total_eve_charge": 22.10,
            "total_night_minutes": 180.0,
            "total_night_calls": 90,
            "total_night_charge": 8.10,
            "total_intl_minutes": 18.0,
            "total_intl_calls": 7,
            "total_intl_charge": 4.86,
            "number_customer_service_calls": 5
        }
    ])
    return sample_data


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("📊 Project Overview")

st.sidebar.markdown("""
## Telecom Churn Intelligence Dashboard

This application helps telecom businesses:

- Predict **customer churn risk**
- Identify **customer segment**
- Recommend **retention strategies**
- Support **data-driven customer decisions**

---

### 🔍 Model Components
- **Churn Prediction Model:** Tuned XGBoost
- **Threshold Used:** 0.60
- **Segmentation Model:** KMeans Clustering (4 Segments)

---

### 🎯 Business Objective
Reduce churn by combining:
- predictive analytics
- customer segmentation
- retention strategy recommendations
""")

st.sidebar.markdown("---")

# Sample CSV download
sample_csv = get_sample_csv().to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    label="⬇️ Download Sample CSV",
    data=sample_csv,
    file_name="sample_telecom_customers.csv",
    mime="text/csv",
    use_container_width=True
)

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit | Portfolio ML Project")

# =========================================================
# MAIN HEADER
# =========================================================
st.markdown('<div class="section-header">📉 Telecom Churn Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown("""
A business-focused machine learning application for **customer churn prediction**, **behavioral segmentation**, and **retention strategy planning**.
""")

st.markdown("---")

# =========================================================
# LOAD MODEL FUNCTIONS SAFELY
# =========================================================
try:
    predict_single_customer, predict_customer = load_prediction_functions()
    st.success("✅ Prediction engine loaded successfully.")
except Exception as e:
    st.error("❌ Failed to load prediction engine.")
    st.exception(e)
    st.stop()

# =========================================================
# HERO KPI STRIP
# =========================================================
k1, k2, k3 = st.columns(3)

with k1:
    st.markdown('<div class="custom-card"><div class="sub-header">🎯 Business Goal</div><div class="highlight-text">Reduce Customer Churn</div><div class="small-note">Use predictive modeling and segmentation to prioritize retention efforts.</div></div>', unsafe_allow_html=True)

with k2:
    st.markdown('<div class="custom-card"><div class="sub-header">🧠 ML Stack</div><div class="highlight-text">XGBoost + KMeans</div><div class="small-note">Combines supervised churn prediction with unsupervised customer segmentation.</div></div>', unsafe_allow_html=True)

with k3:
    st.markdown('<div class="custom-card"><div class="sub-header">📦 App Capabilities</div><div class="highlight-text">Single + Bulk Predictions</div><div class="small-note">Supports both individual customer scoring and batch business decisioning.</div></div>', unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# TABS
# =========================================================
tab1, tab2 = st.tabs(["🔮 Single Customer Prediction", "📂 Bulk CSV Dashboard"])


# =========================================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# =========================================================
with tab1:
    st.subheader("🔮 Single Customer Prediction")
    st.markdown("Enter customer information below to predict churn risk and segment profile.")

    with st.expander("📝 Enter Customer Details", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            account_length = st.number_input("Account Length", min_value=1, value=120)
            international_plan = st.selectbox("International Plan", ["no", "yes"])
            voice_mail_plan = st.selectbox("Voice Mail Plan", ["no", "yes"])
            voice_mail_messages = st.number_input("Voice Mail Messages", min_value=0, value=25)
            number_customer_service_calls = st.number_input(
                "Customer Service Calls", min_value=0, value=2
            )

        with col2:
            total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, value=250.5)
            total_day_calls = st.number_input("Total Day Calls", min_value=0, value=110)
            total_day_charge = st.number_input("Total Day Charge", min_value=0.0, value=42.59)

            total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, value=180.2)
            total_eve_calls = st.number_input("Total Evening Calls", min_value=0, value=95)
            total_eve_charge = st.number_input("Total Evening Charge", min_value=0.0, value=15.32)

        with col3:
            total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0, value=220.4)
            total_night_calls = st.number_input("Total Night Calls", min_value=0, value=100)
            total_night_charge = st.number_input("Total Night Charge", min_value=0.0, value=9.92)

            total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, value=12.5)
            total_intl_calls = st.number_input("Total International Calls", min_value=0, value=4)
            total_intl_charge = st.number_input("Total International Charge", min_value=0.0, value=3.38)

    if st.button("🚀 Predict Customer Outcome", use_container_width=True):
        try:
            customer_data = {
                "account_length": account_length,
                "international_plan": international_plan,
                "voice_mail_plan": voice_mail_plan,
                "voice_mail_messages": voice_mail_messages,

                "total_day_minutes": total_day_minutes,
                "total_day_calls": total_day_calls,
                "total_day_charge": total_day_charge,

                "total_eve_minutes": total_eve_minutes,
                "total_eve_calls": total_eve_calls,
                "total_eve_charge": total_eve_charge,

                "total_night_minutes": total_night_minutes,
                "total_night_calls": total_night_calls,
                "total_night_charge": total_night_charge,

                "total_intl_minutes": total_intl_minutes,
                "total_intl_calls": total_intl_calls,
                "total_intl_charge": total_intl_charge,

                "number_customer_service_calls": number_customer_service_calls
            }

            result = predict_single_customer(customer_data)
            row = result.iloc[0]

            st.success("Prediction completed successfully!")

            st.markdown("## 📊 Prediction Summary")

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            with kpi1:
                st.metric("Churn Probability", f"{row['churn_probability'] * 100:.2f}%")

            with kpi2:
                st.metric("Churn Prediction", "Yes" if row["churn_prediction"] == 1 else "No")

            with kpi3:
                st.metric("Risk Level", row["risk_level"])

            with kpi4:
                st.metric("Cluster ID", int(row["cluster_id"]))

            st.markdown("---")

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### 🚨 Risk Assessment")
                st.info(risk_badge(row["risk_level"]))
                st.progress(min(float(row["churn_probability"]), 1.0))
                st.markdown("### 🎯 Recommended Retention Action")
                st.warning(row["retention_action"])

            with col_b:
                st.markdown("### 🧩 Customer Segment")
                st.success(row["customer_segment"])
                st.markdown("### 📌 Segment-Based Strategy")
                st.info(row["segment_strategy"])

            st.markdown("---")
            st.markdown("## 🧠 Business Interpretation")
            st.markdown(generate_business_summary(row))

            st.markdown("---")
            st.markdown("## 🔍 Full Prediction Output")
            st.dataframe(result, use_container_width=True)

        except Exception as e:
            st.error("❌ Error during single customer prediction.")
            st.exception(e)


# =========================================================
# TAB 2: BULK CSV DASHBOARD
# =========================================================
with tab2:
    st.subheader("📂 Bulk Customer Churn Dashboard")
    st.markdown("Upload a CSV file to generate churn predictions and segment-based business insights for multiple customers.")

    st.download_button(
        label="⬇️ Download Sample CSV Format",
        data=sample_csv,
        file_name="sample_telecom_customers.csv",
        mime="text/csv",
        use_container_width=True
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        st.markdown("### 📄 Uploaded Data Preview")
        st.dataframe(input_df.head(), use_container_width=True)

        if st.button("📈 Run Bulk Prediction", use_container_width=True):
            try:
                result_df = predict_customer(input_df)

                st.success("Bulk prediction completed successfully!")

                total_customers = len(result_df)
                predicted_churners = result_df["churn_prediction"].sum()
                churn_rate = (predicted_churners / total_customers) * 100
                avg_churn_prob = result_df["churn_probability"].mean() * 100

                st.markdown("## 📊 Bulk Prediction Summary")

                m1, m2, m3, m4 = st.columns(4)

                with m1:
                    st.metric("Total Customers", total_customers)

                with m2:
                    st.metric("Predicted Churners", int(predicted_churners))

                with m3:
                    st.metric("Predicted Churn Rate", f"{churn_rate:.2f}%")

                with m4:
                    st.metric("Avg Churn Probability", f"{avg_churn_prob:.2f}%")

                st.markdown("---")

                chart1, chart2 = st.columns(2)

                with chart1:
                    st.markdown("### 📌 Risk Level Distribution")
                    risk_counts = result_df["risk_level"].value_counts()

                    fig, ax = plt.subplots(figsize=(6, 4))
                    risk_counts.plot(kind="bar", ax=ax)
                    ax.set_xlabel("Risk Level")
                    ax.set_ylabel("Count")
                    ax.set_title("Risk Level Distribution")
                    st.pyplot(fig)

                with chart2:
                    st.markdown("### 🧩 Customer Segment Distribution")
                    segment_counts = result_df["customer_segment"].value_counts()

                    fig, ax = plt.subplots(figsize=(6, 4))
                    segment_counts.plot(kind="bar", ax=ax)
                    ax.set_xlabel("Customer Segment")
                    ax.set_ylabel("Count")
                    ax.set_title("Customer Segment Distribution")
                    plt.xticks(rotation=30, ha="right")
                    st.pyplot(fig)

                st.markdown("---")

                st.markdown("## 🚨 High-Risk Customers Requiring Attention")
                high_risk_df = result_df[result_df["risk_level"].isin(["High Risk", "Very High Risk"])]

                if not high_risk_df.empty:
                    st.dataframe(high_risk_df, use_container_width=True)
                else:
                    st.success("No high-risk customers identified in this upload.")

                st.markdown("---")

                st.markdown("## 🔍 Full Prediction Results")
                st.dataframe(result_df, use_container_width=True)

                csv = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Predictions as CSV",
                    data=csv,
                    file_name="telecom_churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:
                st.error("❌ Error during bulk prediction.")
                st.exception(e)


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("🚀 Built as an end-to-end ML business solution using Streamlit, XGBoost, and KMeans Clustering.")