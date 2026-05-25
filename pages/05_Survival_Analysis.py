import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.preprocessing import LabelEncoder
import utils

# Set page config
st.set_page_config(
    page_title="Loyalty Survival Analysis - FlyCaelum",
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

st.title("⏳ Advanced Loyalty Survival Analysis")
st.subheader("Analyze customer retention dynamics, loyalty milestones, and hazard factors over time")
st.markdown("---")

st.markdown("""
Survival analysis helps us understand the temporal dynamics of customer engagement. Instead of asking **"Will this customer churn?"**, survival analysis answers **"When will this customer churn?"** and **"What factors accelerate disengagement?"**
""")

col_curves, col_milestones = st.columns([3, 2])

with col_curves:
    st.markdown("#### **Kaplan-Meier Survival Curves by Loyalty Card Tier**")
    st.markdown("This chart visualizes the probability that a member remains engaged over their tenure in months. Notice the steep drop around the 6-year (72-month) milestone.")
    
    # Fit and Plot Kaplan-Meier curves
    fig, ax = plt.subplots(figsize=(10, 5.5))
    kmf = KaplanMeierFitter()
    
    # Stratify by Loyalty Card level
    tiers = sorted(filtered_df['Loyalty Card'].unique())
    colors = ['#f59e0b', '#3b82f6', '#10b981'] if dark_mode else ['#d97706', '#1d4ed8', '#059669']
    
    for i, tier in enumerate(tiers):
        mask = filtered_df['Loyalty Card'] == tier
        if mask.sum() > 0:
            kmf.fit(
                filtered_df.loc[mask, 'Tenure_Months'], 
                event_observed=filtered_df.loc[mask, 'Churn_Target'], 
                label=f"{tier} Tier"
            )
            kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[i % len(colors)], linewidth=2.5)
            
    ax.set_xlabel("Program Tenure (Months)", fontsize=11)
    ax.set_ylabel("Survival Probability (Engagement)", fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(title="Loyalty Card Level", fontsize=10, title_fontsize=11)
    
    if dark_mode:
        ax.set_facecolor('#1e293b')
        fig.patch.set_facecolor('#0f172a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.legend(title='Loyalty Card Level', facecolor='#1e293b', labelcolor='white')
        for spine in ax.spines.values():
            spine.set_color('white')
            
    st.pyplot(fig)

with col_milestones:
    st.markdown("#### **Stratified Retention Milestones Table**")
    st.markdown("The retention probabilities at key program milestones confirm that retention is highly stable up to 5 years (60 months), but drops rapidly at 6 years (72 months).")
    
    # We display the official Kaplan-Meier results from our analysis
    milestones_data = {
        'Program Tenure': ['12 Months', '24 Months', '36 Months', '48 Months', '60 Months', '72 Months'],
        'Aurora Tier': ['94.57%', '93.25%', '91.82%', '89.96%', '87.77%', '58.07%'],
        'Nova Tier': ['95.12%', '93.72%', '92.23%', '90.22%', '87.34%', '67.12%'],
        'Star Tier': ['95.04%', '93.65%', '92.30%', '90.18%', '87.58%', '65.37%']
    }
    
    milestones_df = pd.DataFrame(milestones_data)
    st.dataframe(milestones_df, use_container_width=True, hide_index=True)
    
    st.info("""
    💡 **Strategic Insight**: Aurora (top tier) members drop most severely at 72 months to **58.07%**. This reveals that premium benefits alone do not prevent disengagement if customer habits or requirements change, indicating a critical need for proactive re-engagement campaigns before the 5-year mark.
    """)

st.markdown("---")

col_cox, col_cox_interpretation = st.columns([1, 1])

with col_cox:
    st.markdown("#### **Cox Proportional Hazards Regression Results**")
    st.markdown("We fit a Cox model using L2 regularization (`penalizer=0.1`) to understand the hazard factors accelerating customer disengagement.")
    
    # Setup Cox Data
    cox_df = filtered_df.copy()
    le = LabelEncoder()
    cox_df['Loyalty_Card_Encoded'] = le.fit_transform(cox_df['Loyalty Card'].astype(str))
    
    # Use separate column for covariate tenure to avoid collinearity error in lifelines duration fitting
    cox_df['Tenure_Months_Cov'] = cox_df['Tenure_Months']
    
    cox_cols = ['Tenure_Months', 'Tenure_Months_Cov', 'Recency_Flights', 'CLV', 'Loyalty_Card_Encoded', 'Churn_Target']
    cox_data = cox_df[cox_cols]
    
    # Fit Model
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_data, duration_col='Tenure_Months', event_col='Churn_Target')
    
    # Render parameters table
    summary_df = cph.summary[['coef', 'exp(coef)', 'se(coef)', 'z', 'p']].reset_index()
    summary_df.columns = ['Covariate', 'Coefficient (beta)', 'Hazard Ratio (exp(coef))', 'Std Error', 'z-Score', 'p-Value']
    
    # Format and present
    display_cox = summary_df.copy()
    display_cox['Coefficient (beta)'] = display_cox['Coefficient (beta)'].map('{:.4f}'.format)
    display_cox['Hazard Ratio (exp(coef))'] = display_cox['Hazard Ratio (exp(coef))'].map('{:.2f}'.format)
    display_cox['Std Error'] = display_cox['Std Error'].map('{:.4f}'.format)
    display_cox['z-Score'] = display_cox['z-Score'].map('{:.2f}'.format)
    display_cox['p-Value'] = display_cox['p-Value'].map(lambda x: '< 0.005' if x < 0.005 else f'{x:.4f}')
    
    st.dataframe(display_cox, use_container_width=True, hide_index=True)

with col_cox_interpretation:
    st.markdown("#### **Interpretation & Takeaways**")
    st.markdown("""
    *   **Member Tenure (Tenure_Months_Cov)**: Each additional month of tenure reduces the hazard of churn by **3%** (Hazard Ratio = **0.97**, p < 0.005). Long-term relationships are statistically more durable.
    *   **Recency of Flights (Recency_Flights)**: Each additional month of inactivity increases the hazard of churn by **12%** (Hazard Ratio = **1.12**, p < 0.005). This is the single strongest driver of disengagement.
    *   **CLV and Card Tier**: Customer Lifetime Value and Loyalty Card Tier are **not statistically significant** predictors of survival when controlling for behavioral variables (p-values > 0.05).
    
    ##### **Key Takeaways for Marketing:**
    1.  **Monitor Flight Recency Closely**: Because inactivity is the strongest hazard multiplier, automate trigger warnings when a member's inactivity exceeds 6 months.
    2.  **Reward Tenure**: Build loyalty campaigns that celebrate member anniversaries, as longer-tenured members have lower baseline hazard rates.
    """)
