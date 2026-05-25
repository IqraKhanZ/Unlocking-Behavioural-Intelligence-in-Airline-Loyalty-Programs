import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils

# Set page config
st.set_page_config(
    page_title="Churn Risk Monitoring - FlyCaelum",
    page_icon="✈️",
    layout="wide"
)

# Apply global styling based on theme state
dark_mode = st.session_state.get('dark_mode', False)
utils.apply_custom_theme(dark_mode)

# Load filtered data from session state (with outdated cache safeguard)
if 'filtered_df' in st.session_state and 'BDS_Score' in st.session_state['filtered_df'].columns:
    filtered_df = st.session_state['filtered_df']
else:
    cohort, cfa = utils.load_data()
    st.session_state['filtered_df'] = cohort
    st.session_state['cfa'] = cfa
    filtered_df = cohort

st.title("🔍 Churn Risk Monitoring")
st.subheader("Explore the churn risk profile of our loyalty program cohort")
st.markdown("---")

# Page Content Layout
col_plots, col_drivers = st.columns([1, 1])

with col_plots:
    st.markdown("#### **Churn Risk Probability Distribution**")
    st.markdown("This histogram shows the distribution of predicted churn probabilities across the active customer cohort. Most members are stable, but a distinct group has high risk.")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(filtered_df['Churn_Probability'], bins=20, kde=True, color='#ef4444', ax=ax)
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Count of Members")
    
    # Apply dark mode chart adjustments
    if dark_mode:
        ax.set_facecolor('#1e293b')
        fig.patch.set_facecolor('#0f172a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('white')
            
    st.pyplot(fig)

with col_drivers:
    st.markdown("#### **XGBoost Churn Risk Drivers**")
    st.markdown("These are the most important features driving our churn model's risk scoring. Recent flight inactivity and points accumulation drops dominate the model's logic.")
    
    # Feature importances grounded in our XGBoost findings
    feature_imp = {
        'Recency of Flights (Months)': 0.444683,
        'Points Accumulated (Last 6M)': 0.168429,
        'Member Tenure (Months)': 0.112424,
        'Recency of Redemptions': 0.019490,
        'Flight Count (Last 6M)': 0.018405,
        'Points Velocity': 0.018384,
        'Education Level': 0.017409,
        'Points Accumulated (Last 12M)': 0.017320,
        'Points Redeemed (Last 6M)': 0.016881,
        'Redemption Ratio': 0.016857
    }
    
    fi_df = pd.DataFrame(list(feature_imp.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True)
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    bar_color = '#3b82f6' if dark_mode else '#1e3a8a'
    ax2.barh(fi_df['Feature'], fi_df['Importance'], color=bar_color)
    ax2.set_xlabel("Relative Gain (Feature Importance)")
    
    if dark_mode:
        ax2.set_facecolor('#1e293b')
        fig2.patch.set_facecolor('#0f172a')
        ax2.tick_params(colors='white')
        ax2.xaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')
        for spine in ax2.spines.values():
            spine.set_color('white')
            
    st.pyplot(fig2)

st.markdown("---")
st.markdown("### 📋 High-Risk Member Registry")

# Interactive risk threshold slider
threshold = st.slider("Select Churn Probability Threshold for Target Registry", min_value=0.50, max_value=0.95, value=0.70, step=0.05)

# Filter high-risk customers
high_risk_registry = filtered_df[filtered_df['Churn_Probability'] >= threshold].sort_values(by='Churn_Probability', ascending=False)

col_stats, col_table = st.columns([1, 4])

with col_stats:
    # Summarize high-risk cohort stats
    st.metric("Members in Registry", f"{len(high_risk_registry):,}")
    st.metric("Total Revenue at Risk", f"${high_risk_registry['CLV'].sum():,.2f}")
    st.metric("Average Salary", f"${high_risk_registry['Salary'].mean():,.2f}")

with col_table:
    st.markdown(f"**Target Registry (Churn Probability $\\ge$ {threshold:.0%})**")
    
    # Format table for clean display
    display_df = high_risk_registry[[
        'Loyalty Number', 'Persona', 'Loyalty Card', 'CLV', 'Salary', 'Churn_Probability', 'BDS_Score', 'Churn_Type', 'Retention_Action'
    ]].copy()
    
    # Rename columns for presentation
    display_df.columns = [
        'Loyalty Number', 'Customer Persona', 'Loyalty Card Tier', 'CLV (CAD)', 'Annual Salary (CAD)', 'Churn Probability', 'BDS Score', 'Churn Classification', 'Recommended Campaign'
    ]
    
    # Format floating numbers
    display_df['CLV (CAD)'] = display_df['CLV (CAD)'].map('${:,.2f}'.format)
    display_df['Annual Salary (CAD)'] = display_df['Annual Salary (CAD)'].map('${:,.2f}'.format)
    display_df['Churn Probability'] = display_df['Churn Probability'].map('{:.1%}'.format)
    display_df['BDS Score'] = display_df['BDS Score'].map('{:.2f}'.format)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
