import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="AML Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme Presentation Styles (CSS) ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0a0e1a; color: #e2e8f0; }
.stApp { background-color: #0a0e1a; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; letter-spacing: -0.5px; }
.metric-card { background: linear-gradient(135deg, #111827 0%, #1a2235 100%); border: 1px solid #1e3a5f; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 24px rgba(0,180,255,0.07); }
.metric-value { font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700; background: linear-gradient(90deg, #00b4d8, #0077b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }
.section-header { font-family: 'Space Mono', monospace; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 3px; color: #00b4d8; border-left: 3px solid #00b4d8; padding-left: 10px; margin-bottom: 16px; }
.stSelectbox > div > div { background-color: #111827 !important; border: 1px solid #1e3a5f !important; color: #e2e8f0 !important; }
.stButton > button { background: linear-gradient(135deg, #0077b6, #00b4d8) !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: 'Space Mono', monospace !important; font-weight: 700 !important; padding: 12px 28px !important; letter-spacing: 1px !important; transition: all 0.2s !important; }
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 24px rgba(0,180,216,0.3) !important; }
div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #111827 100%); border-right: 1px solid #1e3a5f; }
.banner { background: linear-gradient(135deg, #0d1b2a 0%, #1a2f4a 50%, #0d1b2a 100%); border: 1px solid #1e3a5f; border-radius: 16px; padding: 32px 40px; margin-bottom: 32px; position: relative; overflow: hidden; }
.tag { display: inline-block; background: rgba(0,180,216,0.12); border: 1px solid rgba(0,180,216,0.3); border-radius: 20px; padding: 3px 12px; font-size: 0.72rem; color: #00b4d8; font-family: 'Space Mono', monospace; letter-spacing: 1px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Operational Global Schemas ─────────────────────────────────
FEATURE_COLUMNS_ORDER = [
    'Amount', 'Payment_currency', 'Received_currency', 'Sender_bank_location',
    'Receiver_bank_location', 'Payment_type', 'Hour_of_Day', 'Amount_Log',
    'Amount_Above_1k', 'Amount_Bin', 'Payment_Type_Risk_Weight', 'Is_Cross_Location',
    'Is_Same_Location', 'Currency_Mismatch', 'Is_Business_Hours', 'Is_Night_Transaction',
    'Is_Weekend', 'Time_Since_Last_Tx', 'Sender_Fan_Out_Count', 'Is_Sender_Also_Receiver',
    'Benford_First_Digit', 'Amount_ZScore', 'Is_IQR_Outlier', 'Isolation_Forest_Score',
    'One_Class_SVM_Score'
]

# ── Dynamic Simulation Engine ─────────────────────────────────
def generate_pipeline_raw_simulation(n_rows: int, fraud_pct: float) -> pd.DataFrame:
    """Generates a raw dataset structure mapping exactly to the notebook's input schema."""
    np.random.seed(42)
    n_fraud = int(n_rows * fraud_pct / 100)
    n_normal = n_rows - n_fraud
    
    amounts = np.concatenate([np.random.exponential(scale=4500, size=n_fraud), np.random.exponential(scale=400, size=n_normal)])
    is_fraud_flag = np.concatenate([np.ones(n_fraud), np.zeros(n_normal)])
    
    df_raw = pd.DataFrame({
        'Time': [f"{np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}:{np.random.randint(0,60):02d}" for _ in range(n_rows)],
        'Date': pd.date_range(start='2026-01-01', periods=n_rows, freq='min').strftime('%Y-%m-%d'),
        'Sender_account': np.random.choice(np.arange(1001, 1080), n_rows),
        'Receiver_account': np.random.choice(np.arange(2001, 2080), n_rows),
        'Amount': np.abs(amounts) + 0.1,
        'Payment_currency': np.random.choice(['USD', 'EUR', 'GBP', 'INR', 'BTC'], n_rows),
        'Received_currency': np.random.choice(['USD', 'EUR', 'GBP', 'INR', 'AED'], n_rows),
        'Sender_bank_location': np.random.choice(['US', 'UK', 'EU', 'IN', 'XM'], n_rows),
        'Receiver_bank_location': np.random.choice(['US', 'UK', 'EU', 'IN', 'KY'], n_rows),
        'Payment_type': np.random.choice(['Credit card', 'Debit card', 'Cheque', 'Cash Deposit', 'Cross-border'], n_rows),
        'Is_laundering': is_fraud_flag.astype(int),
        'Laundering_type': 'Simulation'
    })
    return df_raw

# ── Notebook Engine Feature Pipeline Implementation ─────────────
def execute_notebook_feature_pipeline(df_raw, artifacts, is_training=False):
    """Executes structural feature engineering steps without data leakage or state errors."""
    df = df_raw.copy()
    
    # Standard Date formatting
    df['Combined_DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    df['Combined_DateTime'] = df['Combined_DateTime'].fillna(pd.Timestamp('2026-01-01 12:00:00'))
    df = df.sort_values(by='Combined_DateTime').reset_index(drop=True)
    df['Hour_of_Day'] = df['Combined_DateTime'].dt.hour
    
    # Stateless Feature Engine Block
    df['Amount_Log'] = np.log1p(df['Amount'])
    df['Amount_Above_1k'] = (df['Amount'] > 1000).astype(int)
    df['Amount_Bin'] = pd.cut(df['Amount'], bins=[-1, 449, 1000, 2791, np.inf], labels=[0, 1, 2, 3]).astype(int)
    
    payment_risk_map = {'Cross-border': 0.768, 'Cash Deposit': 0.490, 'Cheque': 0.316, 'Credit card': 0.185, 'Debit card': 0.185}
    df['Payment_Type_Risk_Weight'] = df['Payment_type'].map(payment_risk_map).fillna(0.20)
    
    df['Is_Cross_Location'] = (df['Sender_bank_location'] != df['Receiver_bank_location']).astype(int)
    df['Is_Same_Location'] = (df['Sender_bank_location'] == df['Receiver_bank_location']).astype(int)
    df['Currency_Mismatch'] = (df['Payment_currency'] != df['Received_currency']).astype(int)
    
    df['Is_Business_Hours'] = df['Hour_of_Day'].isin([8, 9, 10, 11, 14, 15]).astype(int)
    df['Is_Night_Transaction'] = ((df['Hour_of_Day'] >= 22) | (df['Hour_of_Day'] < 6)).astype(int)
    df['Is_Weekend'] = df['Combined_DateTime'].dt.weekday.isin([5, 6]).astype(int)
    
    df['Time_Since_Last_Tx'] = df.groupby('Sender_account')['Combined_DateTime'].diff().dt.total_seconds().fillna(-1)
    df['Sender_Fan_Out_Count'] = df.groupby('Sender_account')['Receiver_account'].transform('nunique')
    
    if is_training:
        artifacts["receivers_pool"] = set(df['Receiver_account'].unique())
    pool = artifacts.get("receivers_pool", set())
    df['Is_Sender_Also_Receiver'] = df['Sender_account'].apply(lambda x: 1 if x in pool else 0)
    
    df['Benford_First_Digit'] = df['Amount'].apply(lambda x: int(str(x).replace('.', '').lstrip('0')[0]) if len(str(x).replace('.', '').lstrip('0')) > 0 else 0)
    
    # Text Categorical Label Encoding Transforms (Prevents Data Leakage/Unseen Class Crashes)
    categorical_cols = ['Payment_currency', 'Received_currency', 'Sender_bank_location', 'Receiver_bank_location', 'Payment_type']
    for col in categorical_cols:
        if is_training:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            artifacts[f"le_{col}"] = le
        else:
            le = artifacts.get(f"le_{col}")
            if le:
                current_labels = set(df[col].astype(str).unique())
                known_labels = set(le.classes_)
                unseen = current_labels - known_labels
                if unseen:
                    le.classes = np.append(le.classes_, list(unseen))
                df[col] = le.transform(df[col].astype(str))
            else:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
    # Stateful Pipeline Parameter Matching
    if is_training:
        artifacts["mean_val"] = df['Amount'].mean()
        artifacts["std_val"] = df['Amount'].std() if df['Amount'].std() != 0 else 1
        Q1 = df['Amount'].quantile(0.25)
        Q3 = df['Amount'].quantile(0.75)
        IQR = Q3 - Q1
        artifacts["lower_bound"] = Q1 - 1.5 * IQR
        artifacts["upper_bound"] = Q3 + 1.5 * IQR

    df['Amount_ZScore'] = (df['Amount'] - artifacts["mean_val"]) / artifacts["std_val"]
    df['Is_IQR_Outlier'] = ((df['Amount'] < artifacts["lower_bound"]) | (df['Amount'] > artifacts["upper_bound"])).astype(int)
    
    # Scale Arrays for Anomaly Engines
    anomaly_subset = ['Amount', 'Amount_Log', 'Time_Since_Last_Tx', 'Sender_Fan_Out_Count']
    if is_training:
        scaler = StandardScaler()
        scaler.fit(df[anomaly_subset].fillna(0))
        artifacts["scaler"] = scaler

    scaled_anom = artifacts["scaler"].transform(df[anomaly_subset].fillna(0))
    
    if is_training:
        artifacts["iforest"] = IsolationForest(n_estimators=20, contamination=0.01, random_state=42).fit(scaled_anom)
        artifacts["ocsvm"] = OneClassSVM(nu=0.01, kernel='rbf', max_iter=100).fit(scaled_anom[:2000])

    df['Isolation_Forest_Score'] = artifacts["iforest"].predict(scaled_anom)
    df['One_Class_SVM_Score'] = artifacts["ocsvm"].predict(scaled_anom)
    
    return df

# ── Load or Rebuild Pipeline Infrastructure ───────────────────
@st.cache_resource(show_spinner=False)
def load_production_pipeline_infrastructure(model_selection):
    artifact_path = "aml_pipeline_production_artifacts.pkl"
    
    if os.path.exists(artifact_path):
        artifacts = joblib.load(artifact_path)
        return artifacts, artifacts["rf_model"] if model_selection == "Random Forest" else artifacts["xgb_model"]
    else:
        # Self-Healing Fallback Mechanism: Fully aligned with the functional training parameters
        st.warning("⚠️ Serialized pipeline artifacts not found. Building pipeline structures on startup...")
        np.random.seed(42)
        mock_raw = pd.DataFrame({
            'Time': [f"{np.random.randint(0,24):02d}:00:00" for _ in range(20000)],
            'Date': ['2026-01-01'] * 20000,
            'Sender_account': np.random.randint(1000, 2000, 20000),
            'Receiver_account': np.random.randint(2000, 3000, 20000),
            'Amount': np.random.exponential(scale=600, size=20000),
            'Payment_currency': np.random.choice(['USD', 'EUR', 'GBP'], 20000),
            'Received_currency': np.random.choice(['USD', 'EUR', 'GBP'], 20000),
            'Sender_bank_location': np.random.choice(['US', 'UK', 'EU'], 20000),
            'Receiver_bank_location': np.random.choice(['US', 'UK', 'EU'], 20000),
            'Payment_type': np.random.choice(['Credit card', 'Cash Deposit', 'Cross-border'], 20000),
            'Is_laundering': np.random.choice([0, 1], 20000, p=[0.98, 0.02]),
            'Laundering_type': 'Normal'
        })
        
        b_artifacts = {}
        # Let the operational pipeline extract metrics organically during fallback training
        engineered_mock = execute_notebook_feature_pipeline(mock_raw, b_artifacts, is_training=True)
        X = engineered_mock[FEATURE_COLUMNS_ORDER].fillna(0)
        y = engineered_mock['Is_laundering'].astype(int)
        
        b_artifacts["rf_model"] = RandomForestClassifier(n_estimators=50, max_depth=8, class_weight='balanced', random_state=42).fit(X, y)
        
        neg, pos = len(y) - y.sum(), y.sum()
        pos_weight = (neg / pos) if pos > 0 else 1.0
        b_artifacts["xgb_model"] = XGBClassifier(n_estimators=50, max_depth=4, scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42).fit(X, y)
        
        return b_artifacts, b_artifacts["rf_model"] if model_selection == "Random Forest" else b_artifacts["xgb_model"]

def classify_risk(prob: float) -> str:
    if prob >= 0.65: return "HIGH"
    elif prob >= 0.35: return "MID"
    return "LOW"

# ── Sidebar Control Panel ──────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px 0'>
        <span style='font-family:Space Mono,monospace;font-size:1.1rem;color:#00b4d8;font-weight:700;letter-spacing:2px'>◈ AML ENGINE</span><br>
        <span style='font-size:0.72rem;color:#475569;letter-spacing:1px'>NOTEBOOK-PIPELINE INTEGRATION</span>
    </div> <hr style='border-color:#1e3a5f;margin:12px 0'>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>PRODUCTION CLASSIFIER</div>", unsafe_allow_html=True)
    selected_model = st.selectbox("Pipeline Architecture Model", ["XGBoost", "Random Forest"])
    
    st.markdown("<hr style='border-color:#1e3a5f;margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>NAVIGATION</div>", unsafe_allow_html=True)
    page = st.radio("", ["📂 Upload & Analyze", "🧪 Simulation Mode"], label_visibility="collapsed")

# ── Initialize Framework ──────────────────────────────────────
artifacts, active_model = load_production_pipeline_infrastructure(selected_model)

# ── Header Dashboard Display Banner ────────────────────────────
st.markdown(f"""
<div class='banner'>
    <div class='tag'>PRODUCTION SYSTEM ALIGNED</div>
    <h1 style='margin:0;font-size:2rem;color:#f1f5f9'>AML Notebook Pipeline Interface</h1>
    <p style='color:#64748b;margin:8px 0 0 0;font-size:0.9rem'>
        Processing active frames using pipeline model: <b style='color:#00b4d8'>{selected_model}</b> · Synchronized features output array: 25 Dimensions.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Navigation Flow Control ───────────────────────────────────
if "Upload" in page:
    st.markdown("<div class='section-header'>PRODUCE RUNTIME FILE DATA</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload core CSV transaction logs", type=["csv"])
    
    if uploaded:
        raw_df = pd.read_csv(uploaded)
        
        # Missing column check step
        required = ['Amount', 'Payment_type', 'Sender_bank_location', 'Receiver_bank_location', 'Payment_currency', 'Received_currency', 'Date', 'Time', 'Sender_account', 'Receiver_account']
        missing = [col for col in required if col not in raw_df.columns]
        
        if missing:
            st.error(f"❌ Missing standard pipeline dataset structural metrics: {missing}")
        else:
            processed_df = execute_notebook_feature_pipeline(raw_df, artifacts, is_training=False)
            X_pred = processed_df[FEATURE_COLUMNS_ORDER].fillna(0)
            
            probs = active_model.predict_proba(X_pred)[:, 1]
            processed_df["fraud_probability"] = probs
            processed_df["risk_level"] = [classify_risk(p) for p in probs]
            
            # --- Render KPI Matrix Outputs ---
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='metric-card'><div class='metric-value'>{len(processed_df):,}</div><div class='metric-label'>TOTAL ROWS</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><div class='metric-value'>{(processed_df['risk_level']=='HIGH').sum()}</div><div class='metric-label'>HIGH RISK ALERTS</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><div class='metric-value'>{(processed_df['risk_level']=='MID').sum()}</div><div class='metric-label'>MID RISK WATCH</div></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-card'><div class='metric-value'>{probs.mean()*100:.2f}%</div><div class='metric-label'>MEAN FRAUD RISK SCORE</div></div>", unsafe_allow_html=True)
            
            # --- Visual Analytics Charts Row ---
            st.markdown("<br>", unsafe_allow_html=True)
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("<div class='section-header'>RISK CLASSIFICATION WEIGHT DISTRIBUTION</div>", unsafe_allow_html=True)
                fig_pie = px.pie(processed_df, names='risk_level', color='risk_level',
                                 color_discrete_map={'HIGH': '#ef4444', 'MID': '#f59e0b', 'LOW': '#22c55e'}, hole=0.6)
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_right:
                st.markdown("<div class='section-header'>PIPELINE MODEL LOG PROBABILITIES</div>", unsafe_allow_html=True)
                fig_hist = px.histogram(processed_df, x="fraud_probability", nbins=30, color_discrete_sequence=['#00b4d8'])
                fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
                st.plotly_chart(fig_hist, use_container_width=True)
                
            st.markdown("<div class='section-header'>EXTRACTED SUSPICIOUS TRANSACTIONS</div>", unsafe_allow_html=True)
            flagged = processed_df[processed_df["risk_level"].isin(["HIGH", "MID"])]
            st.dataframe(flagged[['Amount', 'Payment_type', 'Sender_bank_location', 'Receiver_bank_location', 'fraud_probability', 'risk_level']], use_container_width=True)

else:
    st.markdown("<div class='section-header'>SIMULATION ENGINE CONSTRAINTS</div>", unsafe_allow_html=True)
    col_pct, col_size = st.columns(2)
    fraud_pct = col_pct.slider("Target Anomaly Content Ratio (%)", 5, 50, 15)
    sim_size = col_size.select_slider("Log Volume Processing Limits", options=[1000, 2000, 5000, 10000], value=2000)
    
    if st.button("▶ RUN PIPELINE SIMULATION"):
        raw_sim = generate_pipeline_raw_simulation(sim_size, fraud_pct)
        processed_sim = execute_notebook_feature_pipeline(raw_sim, artifacts, is_training=False)
        
        X_sim = processed_sim[FEATURE_COLUMNS_ORDER].fillna(0)
        probs = active_model.predict_proba(X_sim)[:, 1]
        
        processed_sim["fraud_probability"] = probs
        processed_sim["risk_level"] = [classify_risk(p) for p in probs]
        
        # Display operational metric dashboard indices
        tp = ((processed_sim['risk_level'].isin(['HIGH', 'MID'])) & (processed_sim['Is_laundering'] == 1)).sum()
        actual_pos = processed_sim['Is_laundering'].sum()
        detection_rate = (tp / actual_pos * 100) if actual_pos > 0 else 100.0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><div class='metric-value'>{sim_size}</div><div class='metric-label'>SIMULATED LOGS</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-value'>{actual_pos}</div><div class='metric-label'>TRUE FRAUD INSTANCES</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-value'>{processed_sim['risk_level'].value_counts().get('HIGH', 0)}</div><div class='metric-label'>PREDICTED HIGH RISK</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><div class='metric-value'>{detection_rate:.1f}%</div><div class='metric-label'>PIPELINE RECALL (TPR)</div></div>", unsafe_allow_html=True)
        
        # Violin and Flag Plots
        st.markdown("<br>", unsafe_allow_html=True)
        col_v, col_i = st.columns([1.2, 1])
        
        with col_v:
            st.markdown("<div class='section-header'>FRAUD RISK DISTRIBUTION PLOT BY ASSIGNED RISK BAND</div>", unsafe_allow_html=True)
            fig_v = px.violin(processed_sim, y="fraud_probability", x="risk_level", color="risk_level", box=True,
                              color_discrete_map={'HIGH': '#ef4444', 'MID': '#f59e0b', 'LOW': '#22c55e'})
            fig_v.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig_v, use_container_width=True)
            
        with col_i:
            st.markdown("<div class='section-header'>CORE ENGINEERING CRITERIA IMPORTANCES</div>", unsafe_allow_html=True)
            importance_series = pd.Series(active_model.feature_importances_, index=FEATURE_COLUMNS_ORDER).sort_values(ascending=False).head(8)
            fig_b = px.bar(x=importance_series.values, y=importance_series.index, orientation='h', color_discrete_sequence=['#00b4d8'])
            fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_b, use_container_width=True)
            
        st.markdown("<div class='section-header'>REAL-TIME PIPELINE RISK OUTPUT REGISTRY</div>", unsafe_allow_html=True)
        st.dataframe(processed_sim[['Amount', 'Hour_of_Day', 'Isolation_Forest_Score', 'One_Class_SVM_Score', 'fraud_probability', 'risk_level']].sort_values(by="fraud_probability", ascending=False), use_container_width=True)

        # ── ADVANCED RISK CRITERIA & COMPLIANCE ACTION FRAMEWORK ──────────
st.markdown("<br><hr style='border-color:#1e3a5f;margin:24px 0'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>RISK TIER CRITERIA BREAKDOWN & COMPLIANCE ACTIONS</div>", unsafe_allow_html=True)

# 1. Quantify why records were flagged (Rule Evaluation Engine)
# Using the active dataframe from your session (works for processed_df or processed_sim)
active_df = processed_df if 'processed_df' in locals() else (processed_sim if 'processed_sim' in locals() else None)

if active_df is not None and not active_df.empty:
    
    # Calculate rule violations for visual distribution
    total_high = (active_df['risk_level'] == 'HIGH').sum()
    total_mid = (active_df['risk_level'] == 'MID').sum()
    
    # Analyze core criteria triggers within the data
    ml_threshold_flags = (active_df['fraud_probability'] >= 0.65).sum()
    unsupervised_anomaly_flags = ((active_df['Isolation_Forest_Score'] == -1) & (active_df['One_Class_SVM_Score'] == -1)).sum()
    velocity_structuring_flags = ((active_df['Time_Since_Last_Tx'] > 0) & (active_df['Time_Since_Last_Tx'] < 60) & (active_df['Amount_Bin'] >= 2)).sum()
    cross_border_mismatch_flags = ((active_df['Is_Cross_Location'] == 1) & (active_df['Currency_Mismatch'] == 1)).sum()
    benford_violations = (~active_df['Benford_First_Digit'].isin([1, 2, 3])).sum()

    # Create UI columns: Left for the investigative graph, Right for Action Protocols
    col_graph, col_actions = st.columns([1.2, 1])
    
    with col_graph:
        st.markdown("<p style='font-size:0.85rem;color:#64748b;margin-bottom:12px;'>Evaluation of internal data points determining Risk Classifications:</p>", unsafe_allow_html=True)
        
        # Prepare data for the criteria evaluation plot
        criteria_data = pd.DataFrame({
            'Review Criteria': [
                'ML Confidence Threshold (≥65%)', 
                'Dual Unsupervised Anomaly Agreement', 
                'Rapid Velocity Structuring (Gap <60s)', 
                'Cross-Border Currency Layering', 
                'Anomalous Benford Digits (Anti-Fraud)'
            ],
            'Trigger Count': [ml_threshold_flags, unsupervised_anomaly_flags, velocity_structuring_flags, cross_border_mismatch_flags, benford_violations],
            'Severity Context': ['HIGH RISK CORE', 'HIGH/MID BEHAVIORAL', 'MID RISK VELOCITY', 'MID RISK GEOGRAPHIC', 'FORENSIC INVESTIGATION']
        }).sort_values(by='Trigger Count', ascending=True)
        
        # Generate custom themed Plotly horizontal bar chart
        fig_criteria = px.bar(
            criteria_data, 
            x='Trigger Count', 
            y='Review Criteria', 
            color='Severity Context',
            orientation='h',
            color_discrete_map={
                'HIGH RISK CORE': '#ef4444',
                'HIGH/MID BEHAVIORAL': '#f43f5e',
                'MID RISK VELOCITY': '#f59e0b',
                'MID RISK GEOGRAPHIC': '#3b82f6',
                'FORENSIC INVESTIGATION': '#06b6d4'
            },
            template='plotly_dark'
        )
        
        fig_criteria.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis=dict(showgrid=True, gridcolor="#1e3a5f"),
            yaxis=dict(showgrid=False),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0)
        )
        st.plotly_chart(fig_criteria, use_container_width=True)
        
    with col_actions:
        st.markdown("<p style='font-size:0.85rem;color:#64748b;margin-bottom:12px;'>Operational SOP Action Protocols to execute:</p>", unsafe_allow_html=True)
        
        # High Risk Action Card Display
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.06); border: 1px solid #ef4444; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
            <div style="font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #ef4444; font-weight: bold; display: flex; justify-content: space-between;">
                <span>🚨 HIGH RISK ALERTS ({total_high} Flagged)</span>
                <span style="background: #ef4444; color: white; padding: 1px 6px; border-radius: 4px; font-size: 0.7rem;">CRITICAL</span>
            </div>
            <p style="font-size: 0.8rem; color: #cbd5e1; margin: 8px 0 4px 0;"><b>Trigger Criteria:</b> ML output $\ge 0.65$ or multiple unsupervised structural anomalies.</p>
            <ul style="font-size: 0.78rem; color: #94a3b8; margin: 0; padding-left: 18px;">
                <li><b>Immediate Action:</b> Programmatic temporary lock applied to routing gateway.</li>
                <li><b>Regulatory Mandate:</b> Auto-compile & stage a Suspicious Transaction Report (STR / SAR).</li>
                <li><b>Escalation:</b> Push case directly to Tier-2 AML Fraud Compliance Unit.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Mid Risk Action Card Display
        st.markdown(f"""
        <div style="background: rgba(245, 158, 11, 0.06); border: 1px solid #f59e0b; border-radius: 8px; padding: 14px;">
            <div style="font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #f59e0b; font-weight: bold; display: flex; justify-content: space-between;">
                <span>⚠️ MID RISK WATCH ({total_mid} Flagged)</span>
                <span style="background: #f59e0b; color: #0a0e1a; padding: 1px 6px; border-radius: 4px; font-size: 0.7rem;">REVIEW</span>
            </div>
            <p style="font-size: 0.8rem; color: #cbd5e1; margin: 8px 0 4px 0;"><b>Trigger Criteria:</b> ML Output $0.35 - 0.64$, or cross-border network discrepancies.</p>
            <ul style="font-size: 0.78rem; color: #94a3b8; margin: 0; padding-left: 18px;">
                <li><b>Immediate Action:</b> Route transaction to the manual Enhanced Due Diligence (EDD) work queue.</li>
                <li><b>Verification:</b> Issue automated KYC credential renewal token to user.</li>
                <li><b>SLA Windows:</b> Clear transaction automatically if not updated by analyst within 24 hours.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("💡 Please upload data or execute a simulation run above to populate the Criteria Review Matrix.")


    # ── ADVANCED INTERACTIVE AML FORENSIC VISUALIZATION SUITE ──────────
st.markdown("<br><hr style='border-color:#1e3a5f;margin:24px 0'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>INTERACTIVE FORENSIC CHARTS & TRANSACTION TRAJECTORIES</div>", unsafe_allow_html=True)

# Select the operational dataframe dynamically from the active runtime space
active_df = processed_df if 'processed_df' in locals() else (processed_sim if 'processed_sim' in locals() else None)

if active_df is not None and not active_df.empty:
    
    # ── ROW 1: NETWORK TOPOLOGY & MULTI-HOP CASH FLOW FLOW ─────────────────
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.markdown("<div class='section-header'>1. NETWORK CONNECTIONS (HIGH RISK RING SAMPLES)</div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.8rem;color:#64748b;margin-bottom:12px;'>Visualizing directional relationships between accounts for top high-risk incidents.</p>", unsafe_allow_html=True)
        
        # Isolate top alerts to prevent graph overload and maintain clean execution speeds
        network_sample = active_df.sort_values(by='fraud_probability', ascending=False).head(35)
        
        unique_nodes = list(set(network_sample['Sender_account'].astype(str)) | set(network_sample['Receiver_account'].astype(str)))
        
        # Calculate a deterministic circular layout coordinate map for nodes
        num_nodes = len(unique_nodes)
        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
        pos = {node: (np.cos(angle), np.sin(angle)) for node, angle in zip(unique_nodes, angles)}
        
        # Construct line trajectories representing wire transactions (Edges)
        edge_x, edge_y = [], []
        for _, row in network_sample.iterrows():
            x0, y0 = pos[str(row['Sender_account'])]
            x1, y1 = pos[str(row['Receiver_account'])]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#1e3a5f'), hoverinfo='none', mode='lines')
        
        # Construct dots representing accounts (Nodes)
        node_x, node_y, node_text, node_color = [], [], [], []
        for node in unique_nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"Account Node: {node}")
            # Highlight node accent color if it functions across both roles
            is_mule = 1 if int(node) in active_df['Sender_account'].values and int(node) in active_df['Receiver_account'].values else 0
            node_color.append('#ef4444' if is_mule else '#00b4d8')
            
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
            marker=dict(showscale=False, color=node_color, size=14, line=dict(width=2, color='#0a0e1a'))
        )
        
        fig_network = go.Figure(data=[edge_trace, node_trace])
        fig_network.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        st.plotly_chart(fig_network, use_container_width=True)

    with row1_col2:
        st.markdown("<div class='section-header'>2. SANKEY ROUTING (CROSS-BORDER TRAJECTORIES)</div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.8rem;color:#64748b;margin-bottom:12px;'>Tracking financial transit volumes flowing from Sender Countries to Receiver Countries.</p>", unsafe_allow_html=True)
        
        # Group and transform structural routing entities into source-target flows
        flow_df = active_df.groupby(['Sender_bank_location', 'Receiver_bank_location'])['Amount'].sum().reset_index()
        
        all_locations = list(set(flow_df['Sender_bank_location']) | set(flow_df['Receiver_bank_location']))
        loc_map = {loc: idx for idx, loc in enumerate(all_locations)}
        
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=15, line=dict(color="#1e3a5f", width=0.5), label=all_locations, color="#0077b6"),
            link=dict(
                source=flow_df['Sender_bank_location'].map(loc_map),
                target=flow_df['Receiver_bank_location'].map(loc_map),
                value=flow_df['Amount'],
                color="rgba(0, 180, 216, 0.15)"
            )
        )])
        fig_sankey.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_sankey, use_container_width=True)

    # ── ROW 2: MACHINE LEARNING SPACES & TIME GEOMETRY ─────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("<div class='section-header'>3. 3D ML CLUSTER DECISION BOUNDARIES</div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.8rem;color:#64748b;margin-bottom:12px;'>De-mystifying Isolation Forest variables by mapping outliers away from core data clouds.</p>", unsafe_allow_html=True)
        
        # Slice frame to preserve responsive client-side UI orbital panning controls
        cluster_sample = active_df.head(1200)
        
        fig_3d = px.scatter_3d(
            cluster_sample, x='Amount_Log', y='Time_Since_Last_Tx', z='Sender_Fan_Out_Count',
            color='risk_level', color_discrete_map={'HIGH': '#ef4444', 'MID': '#f59e0b', 'LOW': '#22c55e'},
            opacity=0.7, size_max=10
        )
        fig_3d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#1e3a5f", showbackground=False),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#1e3a5f", showbackground=False),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#1e3a5f", showbackground=False)
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with row2_col2:
        st.markdown("<div class='section-header'>4. VELOCITY HEATMAP (TEMPORAL BATCH ANOMALIES)</div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.8rem;color:#64748b;margin-bottom:12px;'>Isolating systematic automated laundering runs executing outside of business hours.</p>", unsafe_allow_html=True)
        
        heatmap_data = active_df.copy()
        heatmap_data['Day_of_Week'] = heatmap_data['Combined_DateTime'].dt.day_name()
        
        # Establish structural sorting bounds for calendar sequencing layout
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        fig_heat = px.density_heatmap(
            heatmap_data, x='Hour_of_Day', y='Day_of_Week', z='fraud_probability',
            histfunc='avg', category_orders={'Day_of_Week': day_order},
            color_continuous_scale=['#0d1117', '#0077b6', '#00b4d8', '#ef4444']
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_colorbar=dict(title="Avg Risk Probs")
        )
        st.plotly_chart(fig_heat, use_container_width=True)

else:
    st.info("💡 Awaiting active dataset extraction array matrix to compute visual structural traces.")

    # ── DYNAMIC RISK PARAMETER CONTROLS ─────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>DYNAMIC RISK APPETITE ENGINE TUNING</div>", unsafe_allow_html=True)

# Interactive sliders allowing the compliance analyst to dynamically redefine risk tiers on the fly
st.markdown("<p style='font-size:0.8rem;color:#64748b;margin-bottom:16px;'>Calibrate the probability boundaries dynamically to balance alert volumes against investigation team capacity.</p>", unsafe_allow_html=True)
slider_col1, slider_col2 = st.columns(2)

with slider_col1:
    high_threshold = st.slider("High-Risk Sensitivity Floor (STR Generation)", min_value=0.50, max_value=0.95, value=0.65, step=0.05)
with slider_col2:
    mid_threshold = st.slider("Mid-Risk Warning Threshold (EDD Queue)", min_value=0.15, max_value=0.49, value=0.35, step=0.05)

# Dynamically re-assign risk classes based on the live slider positions
active_df = processed_df if 'processed_df' in locals() else (processed_sim if 'processed_sim' in locals() else None)

if active_df is not None and not active_df.empty:
    def assign_dynamic_tier(prob):
        if prob >= high_threshold: return "HIGH"
        elif prob >= mid_threshold: return "MID"
        return "LOW"
        
    active_df["risk_level"] = active_df["fraud_probability"].apply(assign_dynamic_tier)

    # ── UPDATED GRAPH COLUMNS ROW ───────────────────────────────────────
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("<div class='section-header'>DYNAMIC RISK SUFFICIENCY & ROUTING DRILL-DOWN</div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.8rem;color:#64748b;margin-bottom:12px;'>Click an outer ring segment to inspect underlying processing channels forming the weight balance.</p>", unsafe_allow_html=True)
        
        # Convert raw Payment_type IDs back to readable strings if they were encoded
        inverse_df = active_df.copy()
        if 'le_Payment_type' in artifacts:
            try:
                inverse_df['Payment_type'] = artifacts['le_Payment_type'].inverse_transform(inverse_df['Payment_type'].astype(int))
            except Exception:
                pass # Fallback to existing structural values if mapping doesn't align
        
        # Generate an interactive multi-level Sunburst Chart
        fig_sunburst = px.sunburst(
            inverse_df, 
            path=['risk_level', 'Payment_type'], 
            values='Amount',
            color='risk_level',
            color_discrete_map={'HIGH': '#ef4444', 'MID': '#f59e0b', 'LOW': '#22c55e'},
            branchvalues="total"
        )
        
        fig_sunburst.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_sunburst, use_container_width=True)
        
    with col_right:
        st.markdown("<div class='section-header'>MODEL RISK DENSITY CURVE WITH COMPLIANCE BOUNDS</div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.8rem;color:#64748b;margin-bottom:12px;'>Real-time analysis showing how changing boundaries impacts systemic workloads.</p>", unsafe_allow_html=True)
        
        # Draw a distribution histogram combined with visual line indicators for thresholds
        fig_density = px.histogram(
            inverse_df, 
            x="fraud_probability", 
            nbins=40, 
            color_discrete_sequence=['#00b4d8'],
            labels={'fraud_probability': 'Fraud Confidence Value'}
        )
        
        # Overlay absolute vertical threshold boundary indicators
        fig_density.add_vline(x=high_threshold, line_dash="dash", line_color="#ef4444", annotation_text="HIGH RISK", annotation_position="top right")
        fig_density.add_vline(x=mid_threshold, line_dash="dash", line_color="#f59e0b", annotation_text="MID RISK", annotation_position="top left")
        
        fig_density.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis=dict(showgrid=True, gridcolor="#1e3a5f"),
            yaxis=dict(showgrid=True, gridcolor="#1e3a5f"),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_density, use_container_width=True)