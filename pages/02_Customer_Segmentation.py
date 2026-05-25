import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils

# Set page config
st.set_page_config(
    page_title="Customer Segmentation - FlyCaelum",
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

st.title("👥 Customer Segmentation & Personas")
st.subheader("Explore the value-risk profiles and characteristics of our customer segments")
st.markdown("---")

# Row 1: Interactive Scatter Plot
st.markdown("#### **Value-vs-Risk Persona Matrix**")
st.markdown("This 2D scatter plot maps customer lifetime value (CLV) against predicted churn probability. High-value customers in the upper-right quadrant are our high-priority retention targets (*Slipping Elites*).")

fig, ax = plt.subplots(figsize=(12, 5))
sns.scatterplot(
    data=filtered_df, 
    x='CLV', 
    y='Churn_Probability', 
    hue='Persona', 
    palette='Set1', 
    alpha=0.7, 
    s=55, 
    ax=ax
)
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='High-Risk Threshold')
ax.set_xlabel("Customer Lifetime Value (CLV in CAD)")
ax.set_ylabel("Predicted Churn Probability")
ax.legend(title='Persona Segment', bbox_to_anchor=(1.02, 1), loc='upper left')

if dark_mode:
    ax.set_facecolor('#1e293b')
    fig.patch.set_facecolor('#0f172a')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.legend(title='Persona Segment', bbox_to_anchor=(1.02, 1), loc='upper left', facecolor='#1e293b', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_color('white')

st.pyplot(fig)

st.markdown("---")

# Row 2: Profile Matrix
st.markdown("#### **Segment Profile Matrix**")
st.markdown("This table summarizes the average behavioral and value metrics across the 5 personas.")

profile_cols = ['CLV', 'Salary', 'Flights_Last_12M', 'Recency_Flights', 'Churn_Probability', 'Net_Points_History']
seg_summary = filtered_df.groupby('Persona')[profile_cols].mean()
seg_counts = filtered_df['Persona'].value_counts()
seg_summary['Count'] = seg_counts

display_summary = seg_summary.copy()
display_summary['CLV'] = display_summary['CLV'].map('${:,.2f}'.format)
display_summary['Salary'] = display_summary['Salary'].map('${:,.2f}'.format)
display_summary['Flights_Last_12M'] = display_summary['Flights_Last_12M'].map('{:.2f}'.format)
display_summary['Recency_Flights'] = display_summary['Recency_Flights'].map('{:.1f} months'.format)
display_summary['Churn_Probability'] = display_summary['Churn_Probability'].map('{:.1%}'.format)
display_summary['Net_Points_History'] = display_summary['Net_Points_History'].map('{:,.0f} pts'.format)
display_summary['Count'] = display_summary['Count'].map('{:,}'.format)

st.dataframe(display_summary, use_container_width=True)

st.markdown("#### **CLV Comparison: Active vs. Churn Cohorts**")
st.markdown("This table compares the average Customer Lifetime Value (CLV) between active and churned loyalty members, demonstrating that high-value history is not a churn prevention shield.")

clv_cohort = filtered_df.groupby('Churn_Target')['CLV'].mean().reset_index()
clv_cohort['Churn_Target'] = clv_cohort['Churn_Target'].map({0: 'Active / Engaged', 1: 'Churned / Disengaged'})
clv_cohort.columns = ['Cohort Status', 'Average CLV (CAD)']
clv_cohort['Average CLV (CAD)'] = clv_cohort['Average CLV (CAD)'].map('${:,.2f}'.format)
st.dataframe(clv_cohort, use_container_width=True, hide_index=True)

st.markdown("---")

# Row 3: Persona Deep-Dive
st.markdown("### 🔍 Persona Deep-Dive & Demographics")

# Persona selector
selected_persona = st.selectbox("Select a Persona to Deep-Dive", options=sorted(filtered_df['Persona'].unique()))

persona_df = filtered_df[filtered_df['Persona'] == selected_persona]

# Narrative for the selected persona
narratives = {
    "At-Risk Elites": {
        "description": "These are formerly high-value loyalty members who have been with the program for a long tenure (avg. 5+ years) but have completely stopped flying (last flight was avg. 23 months ago). Their churn risk is extremely high (~95%).",
        "significance": "Critical. This represents silent customer churn that has already happened or is in the final stages. Interventions here are high-priority reactivation campaigns.",
        "opportunity": "Concierge reactivation call, elite tier extension, or a massive mileage voucher for their historical favorite destination."
    },
    "Loyal Champions": {
        "description": "These are our most active and valuable customers. They fly frequently (~23 times a year), have high CLV, and accumulate massive net point balances (~43,000 pts). Churn risk is extremely low (<2%).",
        "significance": "Primary. This group represents the core recurring revenue engine of our airline loyalty program.",
        "opportunity": "VIP appreciation rewards, lounge passes, partner rewards, and early access to new routes/tier level upgrades."
    },
    "New Members (Onboarding)": {
        "description": "These are brand new signups (average tenure < 2 months) who have not flown yet. Their high recency metric is a placeholder. Churn risk is low because they are in the active onboarding lifecycle.",
        "significance": "Medium. Highly valuable group representing growth and future lifetime value.",
        "opportunity": "Targeted welcome bonuses, first-flight discounts, and step-by-step guides on how to redeem points."
    },
    "Corporate Flyers": {
        "description": "These are high-income professionals (average salary is ~$222k CAD) who travel frequently (~18 times a year) and have a very low churn probability (~3%).",
        "significance": "High. These travelers are sensitive to business class comfort, lounge facilities, and schedule convenience.",
        "opportunity": "Airport premium perks, corporate loyalty discounts, business class upgrade promotions, and credit card point matching."
    },
    "Active Leisure": {
        "description": "These are moderate-value customers who travel consistently (~12 times a year) for vacation or leisure. Their churn probability is low (~4%).",
        "significance": "Medium. Represents consistent discretionary travel bookings.",
        "opportunity": "Seasonal leisure travel promotions, partner hotel deals, car rental vouchers, and points discount redemption offers."
    },
    "Ghost Members": {
        "description": "These are completely inactive accounts with zero flights in the last 12 months, zero net points history, and a tenure of more than 6 months. Their churn risk is extremely high (~99%).",
        "significance": "Campaign Waste. These accounts are functionally dead. Marketing to them yields no return and inflates operational campaign costs.",
        "opportunity": "Exclude from active campaign targeting and remove from marketing lists to minimize budget waste."
    }
}

col_narrative, col_demographics = st.columns([1, 1])

with col_narrative:
    p_info = narratives.get(selected_persona, {"description": "", "significance": "", "opportunity": ""})
    st.markdown(f"#### **Profile: {selected_persona}**")
    st.markdown(f"**Description**: {p_info['description']}")
    st.markdown(f"**Business Significance**: {p_info['significance']}")
    st.markdown(f"**Retention Strategy**: *{p_info['opportunity']}*")
    
    # Quick metrics for the selected persona
    st.metric("Total Members in Segment", f"{len(persona_df):,}")
    st.metric("Segment Revenue Contribution", f"${persona_df['CLV'].sum():,.2f} CAD")

with col_demographics:
    st.markdown(f"#### **Demographic Breakdown: {selected_persona}**")
    
    # 2x2 grid for sub-charts
    fig_dem, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Color palette
    color_palette = "Blues_r" if not dark_mode else "coolwarm"
    
    # 1. Loyalty Card status
    sns.countplot(data=persona_df, x='Loyalty Card', order=sorted(persona_df['Loyalty Card'].unique()), palette=color_palette, ax=axes[0, 0])
    axes[0, 0].set_title("Loyalty Card Tier")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("")
    
    # 2. Education level
    sns.countplot(data=persona_df, y='Education', order=persona_df['Education'].value_counts().index, palette=color_palette, ax=axes[0, 1])
    axes[0, 1].set_title("Education Level")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("")
    
    # 3. Marital status
    sns.countplot(data=persona_df, x='Marital Status', order=persona_df['Marital Status'].value_counts().index, palette=color_palette, ax=axes[1, 0])
    axes[1, 0].set_title("Marital Status")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("")
    
    # 4. Salary distribution
    sns.histplot(persona_df['Salary'], bins=15, kde=True, color='#e67e22', ax=axes[1, 1])
    axes[1, 1].set_title("Salary Distribution")
    axes[1, 1].set_xlabel("Salary (CAD)")
    axes[1, 1].set_ylabel("")
    
    if dark_mode:
        fig_dem.patch.set_facecolor('#0f172a')
        for ax_sub in axes.flat:
            ax_sub.set_facecolor('#1e293b')
            ax_sub.tick_params(colors='white')
            ax_sub.xaxis.label.set_color('white')
            ax_sub.yaxis.label.set_color('white')
            ax_sub.title.set_color('white')
            for spine in ax_sub.spines.values():
                spine.set_color('white')
                
    plt.tight_layout()
    st.pyplot(fig_dem)
