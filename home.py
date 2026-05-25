import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils

def render_page():
    # Apply theme styling based on dark mode state
    dark_mode = st.session_state.get('dark_mode', False)
    utils.apply_custom_theme(dark_mode)

    # Load data
    cohort, cfa = utils.load_data()

    # Page Header
    st.title("🏠 FlyCaelum Executive Home")
    st.subheader("Predictive Behavioral Intelligence & Smart Retention Console")
    st.markdown("---")

    # Global Filters on Home (which automatically update cohort in session state)
    st.sidebar.header("Global Filters")
    provinces = st.sidebar.multiselect("Filter by Province", options=sorted(cohort['Province'].unique()), key="h_prov")
    cards = st.sidebar.multiselect("Filter by Loyalty Card Level", options=sorted(cohort['Loyalty Card'].unique()), key="h_card")
    marital_statuses = st.sidebar.multiselect("Filter by Marital Status", options=sorted(cohort['Marital Status'].unique()), key="h_marital")
    genders = st.sidebar.multiselect("Filter by Gender", options=sorted(cohort['Gender'].unique()), key="h_gender")

    # Filter cohort
    filtered_df = cohort.copy()
    if provinces:
        filtered_df = filtered_df[filtered_df['Province'].isin(provinces)]
    if cards:
        filtered_df = filtered_df[filtered_df['Loyalty Card'].isin(cards)]
    if marital_statuses:
        filtered_df = filtered_df[filtered_df['Marital Status'].isin(marital_statuses)]
    if genders:
        filtered_df = filtered_df[filtered_df['Gender'].isin(genders)]

    # Store filtered results in session state for other subpages
    st.session_state['filtered_df'] = filtered_df
    st.session_state['cfa'] = cfa

    # Render KPI Metric Grid
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        utils.render_kpi_card(
            title="Active Loyalty Members",
            value=f"{len(filtered_df):,}",
            subtitle="Monitored in current cohort"
        )

    with col2:
        mean_risk = filtered_df['Churn_Probability'].mean()
        is_high_risk = mean_risk > 0.15
        utils.render_kpi_card(
            title="Average Churn Risk",
            value=f"{mean_risk:.2%}",
            subtitle="Portfolio risk probability",
            is_red=is_high_risk
        )

    with col3:
        high_risk_members = filtered_df[filtered_df['Churn_Probability'] > 0.50]
        revenue_at_risk = high_risk_members['CLV'].sum()
        utils.render_kpi_card(
            title="Portfolio Value at Risk",
            value=f"${revenue_at_risk:,.2f} CAD",
            subtitle=f"{len(high_risk_members):,} high-risk members",
            is_red=True
        )

    with col4:
        saved_value = revenue_at_risk * 0.15
        utils.render_kpi_card(
            title="Est. Retention Savings",
            value=f"${saved_value:,.2f} CAD",
            subtitle="Based on 15% reactivation rate"
        )

    st.markdown("### Executive Overview")

    # Narrative layout
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
        #### **Strategic Business Opportunity**
        Welcome to the FlyCaelum Loyalty Behavioral Intelligence Dashboard. This platform translates predictive machine learning models into actionable customer retention strategies.
        
        Rather than waiting for customers to officially cancel their membership (which is often a late intervention), our model proactively identifies **"silent churn"** by looking for declines in flight velocity, recency changes, and inactivity markers.
        
        ##### **Key Dashboard Capabilities:**
        - **Churn Risk Monitoring (Page 1)**: Track high-risk loyalty members, distribution of churn probability, and extract the key drivers of disengagement.
        - **Customer Segmentation (Page 2)**: Visualizes our 6 core behavioral personas (Loyal Champions, Corporate Flyers, Active Leisure, New Members (Onboarding), Ghost Members, and At-Risk Elites) to identify marketing opportunities.
        - **Customer Drilldown (Page 3)**: Search individual loyalty accounts, view their profiles, and plot historical flight trends over time.
        - **Smart Campaign Console (Page 4)**: Simulate retention campaign launches and calculate real-time financial ROI on saved customer lifetime value (CLV).
        """)
        elites = filtered_df[filtered_df['Persona'] == "At-Risk Elites"]
        elites_count = len(elites)
        elites_saved = elites_count * elites['CLV'].mean() * 0.15 if elites_count > 0 else 0.0
        st.info(f"💡 **Reactivation Highlight**: Slipping Elite members represent a critical chunk of our revenue at risk. Successfully retaining just 15% of this group recovers **${elites_saved:,.2f} CAD** in customer lifetime value.")

    with col_right:
        # Render the Persona distribution plot
        st.write("**Customer Persona Distribution**")
        persona_counts = filtered_df['Persona'].value_counts()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        # Customize colors for dark mode vs light mode compatibility
        plot_color = '#1e3a8a' if not dark_mode else '#3b82f6'
        sns.barplot(x=persona_counts.values, y=persona_counts.index, color=plot_color, ax=ax)
        ax.set_xlabel("Number of Members")
        ax.set_ylabel("")
        
        # If dark mode, style the chart axes text as white
        if dark_mode:
            ax.set_facecolor('#1e293b')
            fig.patch.set_facecolor('#0f172a')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('white')
            
        st.pyplot(fig)

    st.markdown("---")
    st.caption("✈️ FlyCaelum Loyalty Behavioral Intelligence Platform | Powered by XGBoost & KMeans | Data: 2012-2018")
