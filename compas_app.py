
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="COMPAS Fairness Simulator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-score {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-low { color: #059669; }
    .risk-medium { color: #D97706; }
    .risk-high { color: #DC2626; }
    .info-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3B82F6;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚖️ COMPAS Algorithm Fairness Simulator</h1>', unsafe_allow_html=True)
st.markdown("""
**Using actual COMPAS data from cox-parsed.csv**  
*Explore how algorithmic bias affects risk assessment scores*
""")

# Load data (cached for performance)
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv')
    
    # Apply preprocessing
    df['is_high_risk'] = df['score_text'].apply(lambda x: 1 if x == 'High' else 0)
    
    # Clean crime description
    df['crime_category'] = df['c_charge_desc'].fillna('Unknown')
    df['crime_category'] = df['crime_category'].apply(
        lambda x: x.split('(')[0].strip() if '(' in str(x) else str(x)
    )
    
    # Simplify to top categories
    top_crimes = df['crime_category'].value_counts().head(15).index.tolist()
    df['crime_category'] = df['crime_category'].apply(
        lambda x: x if x in top_crimes else 'Other'
    )
    
    # Handle missing columns
    if 'juv_fel_count' not in df.columns:
        df['juv_fel_count'] = 0
    if 'sex' not in df.columns:
        df['sex'] = 'Male'
    
    return df

@st.cache_resource
def get_model(df):
    features = ['age', 'priors_count', 'juv_fel_count', 'race', 'sex', 'crime_category']
    available_features = [f for f in features if f in df.columns]
    
    df_model = df[available_features + ['is_high_risk', 'decile_score']].dropna()
    
    X = pd.get_dummies(df_model[available_features], 
                       columns=[f for f in ['race', 'sex', 'crime_category'] if f in available_features], 
                       drop_first=True)
    y = df_model['is_high_risk']
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X, y)
    
    return model, X.columns.tolist(), df_model

# Load data and model
df = load_data()
model, feature_cols, df_model = get_model(df)

# Sidebar
with st.sidebar:
    st.markdown("## 👤 Defendant Profile")
    
    # Get unique values from actual data
    min_age = int(df['age'].min()) if not df['age'].isnull().all() else 18
    max_age = int(df['age'].max()) if not df['age'].isnull().all() else 70
    age = st.slider("Age", min_age, max_age, 30)
    
    max_priors = int(df['priors_count'].max()) if not df['priors_count'].isnull().all() else 20
    priors = st.slider("Prior Convictions", 0, max_priors, 2)
    
    juv_fel = st.slider("Juvenile Felonies", 0, 5, 0)
    
    # Get race options from actual data
    race_options = sorted([r for r in df['race'].dropna().unique() if isinstance(r, str)])
    race = st.selectbox("Race", race_options)
    
    sex_options = sorted([s for s in df['sex'].dropna().unique() if isinstance(s, str)])
    sex = st.selectbox("Sex", sex_options)
    
    crime_options = sorted([c for c in df['crime_category'].dropna().unique() if isinstance(c, str)])
    crime = st.selectbox("Charge Category", crime_options)
    
    algorithm_mode = st.radio(
        "Algorithm Version",
        ["Real COMPAS (with bias)", "'Fair' Algorithm (bias-corrected)", "Compare Both"]
    )
    
    calculate = st.button("Calculate Risk Score", type="primary", use_container_width=True)

# Prediction function
def predict_risk(age, priors, juv_fel, race, sex, crime, mode='real'):
    # Create input dataframe
    input_data = {
        'age': [age],
        'priors_count': [priors],
        'juv_fel_count': [juv_fel],
        'race': [race],
        'sex': [sex],
        'crime_category': [crime]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # One-hot encode categorical variables
    categorical_cols = ['race', 'sex', 'crime_category']
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    # Ensure all feature columns are present
    for col in feature_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded[feature_cols]
    
    # Predict probability
    prob = model.predict_proba(input_encoded)[0][1]
    
    # Apply fairness adjustments if requested
    if mode == 'fair':
        # Bias correction based on ProPublica findings
        if race == 'African-American':
            prob = prob * 0.7  # Reduce probability for Black defendants
        elif race == 'Caucasian':
            prob = prob * 1.1  # Slight increase for White defendants
        
        # Additional correction for certain crimes
        crime_str = str(crime)
        if any(keyword in crime_str for keyword in ['Drug', 'Battery', 'Possession']):
            if race in ['African-American', 'Hispanic']:
                prob = prob * 0.8
    
    # Convert to decile score (1-10)
    decile = min(10, max(1, int(prob * 10) + 1))
    
    # Determine risk category
    if decile <= 3:
        risk_text = "Low"
    elif decile <= 7:
        risk_text = "Medium"
    else:
        risk_text = "High"
    
    return risk_text, decile, prob

# Main content area
if calculate:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("---")
        
        if "Compare" in algorithm_mode:
            # Compare both algorithms
            risk_real, decile_real, prob_real = predict_risk(
                age, priors, juv_fel, race, sex, crime, 'real'
            )
            risk_fair, decile_fair, prob_fair = predict_risk(
                age, priors, juv_fel, race, sex, crime, 'fair'
            )
            
            # Display comparison
            st.markdown("## 📊 COMPAS Score Comparison")
            
            col_real, col_fair = st.columns(2)
            
            with col_real:
                st.markdown("### Real COMPAS")
                st.markdown(f'<div class="risk-score risk-{risk_real.lower()}">{decile_real}/10</div>', 
                          unsafe_allow_html=True)
                st.markdown(f"**{risk_real} Risk**")
                st.metric("Probability", f"{prob_real:.1%}")
            
            with col_fair:
                st.markdown("### Bias-Corrected")
                st.markdown(f'<div class="risk-score risk-{risk_fair.lower()}">{decile_fair}/10</div>', 
                          unsafe_allow_html=True)
                st.markdown(f"**{risk_fair} Risk**")
                st.metric("Probability", f"{prob_fair:.1%}", 
                         delta=f"{(prob_fair - prob_real):+.1%}")
            
            # Show difference
            diff = decile_real - decile_fair
            if diff > 0:
                st.info(f"📉 Bias correction lowered score by {diff} points")
            elif diff < 0:
                st.warning(f"📈 Bias correction raised score by {abs(diff)} points")
            else:
                st.info("⚖️ No change in score after bias correction")
                
        else:
            # Single algorithm mode
            is_fair = "'Fair' Algorithm" in algorithm_mode
            risk, decile, prob = predict_risk(
                age, priors, juv_fel, race, sex, crime, 
                'fair' if is_fair else 'real'
            )
            
            st.markdown("## 🎯 Your COMPAS Risk Assessment")
            
            # Display score
            st.markdown(f'<div class="risk-score risk-{risk.lower()}">{decile}/10</div>', 
                       unsafe_allow_html=True)
            
            # Progress bar
            st.progress(decile/10, f"Decile Score: {decile}/10")
            
            # Metrics
            col_prob, col_risk = st.columns(2)
            with col_prob:
                st.metric("Recidivism Probability", f"{prob:.1%}")
            with col_risk:
                st.metric("Risk Category", risk)
        
        # Contextual analysis
        st.markdown("---")
        st.markdown("## 📝 Analysis & Context")
        
        if race == 'African-American':
            with st.expander("⚠️ Racial Disparity Alert", expanded=True):
                st.markdown("""
                **Based on ProPublica's analysis of COMPAS:**
                - **44.9%** of Black defendants were falsely labeled high-risk (false positives)
                - **23.5%** of White defendants were falsely labeled high-risk
                - **Gap: 21.4 percentage points**
                
                Black defendants were nearly twice as likely as White defendants to be 
                falsely labeled as high risk of recidivism.
                """)
        
        if 'Drug' in str(crime) and race in ['African-American', 'Hispanic']:
            with st.expander("💊 Drug Offense Context", expanded=True):
                st.markdown("""
                **Historical disparities in drug offenses:**
                - Arrest rates for drug offenses are higher in minority communities
                - Sentencing disparities exist for similar drug offenses
                - These disparities may be reflected in algorithmic risk assessments
                """)
    
    with col2:
        # Statistics and visualizations
        st.markdown("## 📈 Data Insights")
        
        # Calculate actual statistics from your data
        if 'race' in df_model.columns and 'is_high_risk' in df_model.columns:
            # Calculate high risk rates by race
            hr_rates = {}
            for r in ['African-American', 'Caucasian', 'Hispanic']:
                if r in df_model['race'].values:
                    subset = df_model[df_model['race'] == r]
                    if len(subset) > 0:
                        high_risk_rate = subset['is_high_risk'].mean() * 100
                        hr_rates[r] = high_risk_rate
            
            # Create gauge for disparity
            if 'African-American' in hr_rates and 'Caucasian' in hr_rates:
                disparity = hr_rates.get('African-American', 0) - hr_rates.get('Caucasian', 0)
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=disparity,
                    title={"text": "High Risk Rate Gap<br>(Black vs White)"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 40]},
                        'bar': {'color': "red"},
                        'steps': [
                            {'range': [0, 10], 'color': "lightgreen"},
                            {'range': [10, 20], 'color': "yellow"},
                            {'range': [20, 40], 'color': "red"}
                        ],
                    }
                ))
                fig_gauge.update_layout(height=250)
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Histogram of scores by race
        if 'race' in df_model.columns and 'decile_score' in df_model.columns:
            fig_hist = px.histogram(
                df_model, 
                x='decile_score', 
                color='race',
                barmode='overlay',
                title='COMPAS Scores Distribution by Race',
                labels={'decile_score': 'Decile Score', 'race': 'Race'},
                opacity=0.7,
                nbins=10
            )
            fig_hist.update_layout(height=300, bargap=0.1)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Key statistics box
        st.markdown("### 🔍 Key Statistics")
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **ProPublica Findings (2016):**
        - Black false positive rate: **44.9%**
        - White false positive rate: **23.5%**
        - Disparity gap: **21.4 percentage points**
        
        Black defendants were nearly **twice as likely** to be falsely labeled high-risk.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # Show welcome message when not calculated yet
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the COMPAS Fairness Simulator
        
        This tool demonstrates how algorithmic bias can affect risk assessment scores in the criminal justice system.
        
        ### How to use:
        1. Adjust the parameters in the sidebar to create a defendant profile
        2. Choose which algorithm version to compare
        3. Click "Calculate Risk Score" to see the results
        
        ### What you'll learn:
        - How different factors influence risk scores
        - The racial disparities found in COMPAS
        - How bias correction can affect outcomes
        """)
    
    with col2:
        st.markdown("## 📈 Quick Facts")
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **Dataset Summary:**
        - Total records: {:,}
        - Black defendants: {:.1%}
        - White defendants: {:.1%}
        - Average decile score: {:.1f}
        """.format(
            len(df),
            (df['race'] == 'African-American').mean(),
            (df['race'] == 'Caucasian').mean(),
            df['decile_score'].mean() if 'decile_score' in df.columns else 5.5
        ))
        st.markdown('</div>', unsafe_allow_html=True)

# Dataset information in expander
with st.expander("📁 Dataset Information"):
    st.markdown(f"""
    **Dataset:** COMPAS Two-Year Recidivism Data
    **Source:** ProPublica GitHub
    **Total records:** {len(df):,}
    **Records after filtering:** {len(df_model):,}
    
    **Columns used:**
    - Demographic: age, race, sex
    - Criminal history: priors_count, juv_fel_count
    - Charge: crime_category
    - Target: is_high_risk, decile_score
    """)
    
    # Show sample data
    if st.checkbox("Show sample data"):
        st.dataframe(df_model.head(10))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280;">
    <p><small>
    Based on ProPublica's analysis of the COMPAS algorithm (2016)<br>
    <strong>Note:</strong> This is an educational simulation demonstrating algorithmic bias.
    Not an actual risk assessment tool.
    </small></p>
    <p><small>
    <a href="https://github.com/propublica/compas-analysis" target="_blank">View original dataset</a> | 
    <a href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing" target="_blank">Read ProPublica article</a>
    </small></p>
</div>
""", unsafe_allow_html=True)
