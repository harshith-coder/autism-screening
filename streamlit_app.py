import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🧠 Autism Screening | AI-Powered Explainability",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PROFESSIONAL CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f8f9fa;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 2.2em;
        font-weight: 900;
        margin: 10px 0;
    }
    
    .risk-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .risk-percentage {
        font-size: 3.5em;
        font-weight: 900;
        margin: 15px 0;
    }
    
    .danger-box {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_models():
    try:
        with open('models/rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/le_dict.pkl', 'rb') as f:
            le_dict = pickle.load(f)
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('models/shap_explainer.pkl', 'rb') as f:
            explainer = pickle.load(f)
        
        return model, scaler, le_dict, feature_names, explainer
    except Exception as e:
        st.error(f" ❌ Error loading models: {str(e)}")
        return None, None, None, None, None

model, scaler, le_dict, feature_names, explainer = load_models()

if model is None:
    st.error("❌ Models not loaded")
    st.stop()

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.8em;">🧠 Autism Spectrum Screening</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.95;">
            AI-Powered with SHAP Explainability
        </p>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home",
    "📋 Screening",
    "📊 Results",
    "🔍 SHAP",
    "ℹ️ Info"
])

# ============================================================================
# TAB 1: HOME
# ============================================================================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 👋 Welcome to Autism Screening System
        
        This professional AI application helps with early detection of 
        Autism Spectrum Disorder using machine learning.
        
        #### 🎯 What You Can Do:
        - ✅ Complete comprehensive screening questionnaire
        - ✅ Get instant AI-powered risk assessment
        - ✅ Understand predictions via SHAP explainability
        - ✅ Visualize feature contributions
        """)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div>Training Data</div>
            <div class="metric-value">704</div>
        </div>
        <div class="metric-card">
            <div>Accuracy</div>
            <div class="metric-value">92.5%</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: SCREENING FORM
# ============================================================================
with tab2:
    st.markdown("### 📋 Autism Spectrum Quotient Assessment")
    
    with st.form("screening_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Questions 1-5**")
            a1 = st.slider("1. Prefer details over big picture", 0, 1, 0) 
            a2 = st.slider("2. Need sameness and routine", 0, 1, 0)
            a3 = st.slider("3. Prefer systematic reading", 0, 1, 0)
            a4 = st.slider("4. Feel anxious in social situations", 0, 1, 0)
            a5 = st.slider("5. Prefer one-to-one conversations", 0, 1, 0)
        
        with col2:
            st.markdown("**Questions 6-10**")
            a6 = st.slider("6. Notice small environmental changes", 0, 1, 0)
            a7 = st.slider("7. Trouble focusing on transitions", 0, 1, 0)
            a8 = st.slider("8. Often daydream", 0, 1, 0)
            a9 = st.slider("9. Can focus intensely on one topic", 0, 1, 0)
            a10 = st.slider("10. Difficult with small talk", 0, 1, 0)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            gender = st.selectbox("Gender", ["M", "F"])
        with col2:
            ethnicity = st.selectbox("Ethnicity", ["White", "Asian", "Black", "Others"])
            jundice = st.selectbox("Jaundice History", ["no", "yes"])
        with col3:
            autism_family = st.selectbox("Family Autism History", ["no", "yes"])
            country = st.selectbox("Country", ["USA", "UK", "Canada", "India"])
        
        used_app = st.selectbox("Used App Before", ["no", "yes"])
        screening_type = st.selectbox("Screening Type", ["Questionnaire", "Interview"])
        
        if st.form_submit_button("🔍 Get Assessment", use_container_width=True):
            try:
                input_data = {
                    'A1_prefer_detail_not_big_picture': a1,
                    'A2_must_have_sameness': a2,
                    'A3_prefer_reading_systematically': a3,
                    'A4_feel_anxious_in_social': a4,
                    'A5_prefer_talking_one_to_one': a5,
                    'A6_notice_small_changes': a6,
                    'A7_trouble_focus_on_changing': a7,
                    'A8_often_daydream': a8,
                    'A9_focused_on_one_topic': a9,
                    'A10_difficult_small_talk': a10,
                    'age': age,
                    'gender': gender,
                    'ethnicity': ethnicity,
                    'jundice': jundice,
                    'autism_family_member': autism_family,
                    'country': country,
                    'used_app_before': used_app,
                    'screening_type': screening_type
                }
                
                input_df = pd.DataFrame([input_data])
                
                # Encode categorical variables
                input_encoded = input_df.copy()
                for col in le_dict.keys():
                    if col in input_encoded.columns:
                        try:
                            input_encoded[col] = le_dict[col].transform(input_encoded[col])
                        except ValueError:
                            val = input_encoded[col].values[0]
                            valid_classes = list(le_dict[col].classes_)
                            matched = None
                            for vc in valid_classes:
                                if str(val).lower() in str(vc).lower() or str(vc).lower() in str(val).lower():
                                    matched = vc
                                    break
                            if matched:
                                input_encoded[col] = le_dict[col].transform([matched])[0]
                            else:
                                input_encoded[col] = le_dict[col].transform([valid_classes[0]])[0]
                
                # Scale numeric features IN EXACT SCALER ORDER
                # Scaler expects: A1-A10 first, then age (NOT age first!)
                numeric_cols = [c for c in feature_names if c.startswith('A')] + ['age']
                input_scaled = input_encoded.copy()
                input_scaled[numeric_cols] = scaler.transform(input_encoded[numeric_cols])
                
                # Select features in EXACT order as training
                input_final = input_scaled[feature_names].copy()
                
                pred_proba = model.predict_proba(input_final)[0]
                autism_risk = pred_proba[1]
                
                st.session_state.autism_risk = autism_risk
                st.session_state.pred_proba = pred_proba
                st.session_state.input_final = input_final
                
                st.success("✅ Assessment complete! Check Results tab.")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================================================
# TAB 3: RESULTS
# ============================================================================
with tab3:
    if 'autism_risk' not in st.session_state:
        st.info("👈 Complete screening form first")
    else:
        autism_risk = st.session_state.autism_risk
        pred_proba = st.session_state.pred_proba
        
        if autism_risk >= 0.7:
            risk_level = "🔴 HIGH RISK"
            risk_color = "risk-high"
        elif autism_risk >= 0.5:
            risk_level = "🟡 MEDIUM RISK"
            risk_color = "risk-medium"
        else:
            risk_level = "🟢 LOW RISK"
            risk_color = "risk-low"
        
        st.markdown(f"""
        <div class="risk-box {risk_color}">
            <div class="risk-percentage">{autism_risk*100:.1f}%</div>
            <div style="font-size: 1.5em; margin-top: 10px;">{risk_level}</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Autism Risk", f"{autism_risk*100:.1f}%")
        with col2:
            st.metric("No Autism", f"{pred_proba[0]*100:.1f}%")
        with col3:
            st.metric("Confidence", f"{max(pred_proba)*100:.1f}%")
        with col4:
            st.metric("Status", "🏥 Consult MD" if autism_risk >= 0.6 else "✅ Monitor")
        
        st.markdown("---")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(['No Autism', 'Autism'], pred_proba, color=['#00d4ff', '#ff6b6b'], alpha=0.8)
        ax.set_ylim([0, 1])
        for i, v in enumerate(pred_proba):
            ax.text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
        ax.set_title('Risk Assessment', fontweight='bold')
        st.pyplot(fig)

# ============================================================================
# TAB 4: SHAP EXPLANATIONS
# ============================================================================
with tab4:
    if 'autism_risk' not in st.session_state:
        st.info("👈 Complete screening form first")
    else:
        st.markdown("### 🔍 SHAP Feature Importance")
        
        try:
            input_final = st.session_state.input_final
            
            shap_vals = explainer.shap_values(input_final)
            shap_vals_class1 = shap_vals[:, :, 1][0]
            
            feature_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP Value': np.abs(shap_vals_class1)
            }).sort_values('SHAP Value', ascending=True).tail(10)
            
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.barh(feature_imp_df['Feature'], feature_imp_df['SHAP Value'], color='#667eea')
            ax.set_xlabel('|SHAP Value|', fontweight='bold')
            ax.set_title('Top 10 Important Features', fontweight='bold')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================================================
# TAB 5: INFORMATION
# ============================================================================
with tab5:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 About ASD")
        st.markdown("""
        **Autism Spectrum Disorder (ASD)** is a neurodevelopmental condition 
        characterized by:
        
        - Unique social communication patterns
        - Restricted/repetitive behaviors and interests
        - Sensory processing differences
        """)
    
    with col2:
        st.markdown("### 🤖 Model Info")
        st.markdown("""
        - **Algorithm**: Random Forest
        - **Training Data**: 704 samples
        - **Features**: 18
        - **Accuracy**: 92.5%
        - **Explainability**: SHAP
        """)
    
    st.markdown("---")
    st.markdown("""
    <div class="danger-box">
    ⚠️ <strong>DISCLAIMER:</strong> This tool is for screening only, NOT for clinical diagnosis. 
    Always consult qualified healthcare professionals.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.9em;">
🧠 Autism Spectrum Disorder Screening System v1.0
</div>
""", unsafe_allow_html=True)
