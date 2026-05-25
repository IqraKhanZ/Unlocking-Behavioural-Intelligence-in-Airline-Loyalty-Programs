import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils

# Set page config
st.set_page_config(
    page_title="Customer Drilldown - FlyCaelum",
    page_icon="✈️",
    layout="wide"
)

# Apply global styling based on theme state
dark_mode = st.session_state.get('dark_mode', False)
utils.apply_custom_theme(dark_mode)

# Load filtered cohort and activity data (with outdated cache safeguard)
if 'filtered_df' in st.session_state and 'cfa' in st.session_state and 'BDS_Score' in st.session_state['filtered_df'].columns:
    filtered_df = st.session_state['filtered_df']
    cfa = st.session_state['cfa']
else:
    cohort, cfa_raw = utils.load_data()
    st.session_state['filtered_df'] = cohort
    st.session_state['cfa'] = cfa_raw
    filtered_df = cohort
    cfa = cfa_raw

st.title("🔍 Customer Drill-down & Search")
st.subheader("Look up individual customer profiles, demographics, and flight histories")
st.markdown("---")

# Search bar layout
col_search, col_helper = st.columns([1, 2])

with col_search:
    st.markdown("**Search Loyalty Number**")
    
    # Sample numbers
    sample_risk = filtered_df[filtered_df['Churn_Probability'] > 0.80]['Loyalty Number'].head(3).tolist()
    sample_champions = filtered_df[filtered_df['Persona'] == 'Loyal Champions']['Loyalty Number'].head(3).tolist()
    
    st.caption(f"Samples - High Churn Risk: `{sample_risk}`")
    st.caption(f"Samples - Loyal Champions: `{sample_champions}`")
    
    search_id_str = st.text_input("Enter Loyalty Number:", value=str(sample_risk[0]) if sample_risk else "100018")

# Validate lookup
if search_id_str:
    try:
        search_id = int(search_id_str.strip())
        customer_row = filtered_df[filtered_df['Loyalty Number'] == search_id]
        
        if not customer_row.empty:
            customer = customer_row.iloc[0]
            st.markdown(f"### **Customer Account: ID {search_id}**")
            
            # Row 1: Profile Cards
            col_demo, col_predict = st.columns(2)
            
            # Dark mode specific colors for HTML cards
            card_bg = "#1e293b" if dark_mode else "white"
            text_color = "#f8fafc" if dark_mode else "#1f2937"
            border_bottom = "1px solid #334155" if dark_mode else "1px solid #f3f4f6"
            lbl_color = "#94a3b8" if dark_mode else "#4b5563"
            
            with col_demo:
                st.markdown("#### **Demographic Profile**")
                
                html_demo = f"""
                <div style="background-color: {card_bg}; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #1e3a8a; color: {text_color};">
                    <table style="width:100%; border-collapse: collapse; font-size: 15px;">
                        <tr style="border-bottom: {border_bottom}; height: 35px;">
                            <td style="font-weight: 600; color: {lbl_color}; width: 40%;">Loyalty Card Tier</td>
                            <td style="font-weight: 700; color: {text_color};">{customer['Loyalty Card']}</td>
                        </tr>
                        <tr style="border-bottom: {border_bottom}; height: 35px;">
                            <td style="font-weight: 600; color: {lbl_color};">Annual Salary</td>
                            <td style="font-weight: 700; color: {text_color};">${customer['Salary']:,.2f} CAD</td>
                        </tr>
                        <tr style="border-bottom: {border_bottom}; height: 35px;">
                            <td style="font-weight: 600; color: {lbl_color};">Education Level</td>
                            <td style="font-weight: 700; color: {text_color};">{customer['Education']}</td>
                        </tr>
                        <tr style="border-bottom: {border_bottom}; height: 35px;">
                            <td style="font-weight: 600; color: {lbl_color}; font-size: 15px;">Marital Status</td>
                            <td style="font-weight: 700; color: {text_color};">{customer['Marital Status']}</td>
                        </tr>
                        <tr style="border-bottom: {border_bottom}; height: 35px;">
                            <td style="font-weight: 600; color: {lbl_color};">Gender</td>
                            <td style="font-weight: 700; color: {text_color};">{customer['Gender']}</td>
                        </tr>
                        <tr style="height: 35px;">
                            <td style="font-weight: 600; color: {lbl_color};">Program Tenure</td>
                            <td style="font-weight: 700; color: {text_color};">{customer['Tenure_Months']} months</td>
                        </tr>
                    </table>
                </div>
                """
                st.markdown(html_demo, unsafe_allow_html=True)
                
            with col_predict:
                st.markdown("#### **Value & Predictive Insights**")
                
                risk_color = "#ef4444" if customer['Churn_Probability'] > 0.50 else "#10b981"
                html_predict = f"""
                <div style="background-color: {card_bg}; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid {risk_color}; color: {text_color};">
                    <table style="width:100%; border-collapse: collapse; font-size: 15px;">
                        <tr style="border-bottom: {border_bottom}; height: 35px;">
                            <td style="font-weight: 600; color: {lbl_color}; width: 40%;">Customer Persona</td>
                            <td style="font-weight: 700; color: {text_color};">{customer['Persona']}</td>
                        </tr>
                        <tr style="border-bottom: {border_bottom}; height: 35px;">
                            <td style="font-weight: 600; color: {lbl_color};">Lifetime Value (CLV)</td>
                            <td style="font-weight: 700; color: {text_color};">${customer['CLV']:,.2f} CAD</td>
                        </tr>
                        <tr style="border-bottom: {border_bottom}; height: 35px;">
                            <td style="font-weight: 600; color: {lbl_color};">Churn Risk Score</td>
                            <td style="font-weight: 700; color: {risk_color}; font-size: 18px;">{customer['Churn_Probability']:.2%}</td>
                        </tr>
                        <tr style="border-bottom: {border_bottom}; height: 35px;">
                            <td style="font-weight: 600; color: {lbl_color};">Disengagement (BDS)</td>
                            <td style="font-weight: 700; color: {text_color};">{customer['BDS_Score']:.2f}</td>
                        </tr>
                        <tr style="border-bottom: {border_bottom}; height: 35px;">
                            <td style="font-weight: 600; color: {lbl_color};">Churn Classification</td>
                            <td style="font-weight: 700; color: {risk_color};">{customer['Churn_Type']}</td>
                        </tr>
                        <tr style="height: 35px;">
                            <td style="font-weight: 600; color: {lbl_color};">Retention Campaign</td>
                            <td style="font-weight: 700; color: #3b82f6;">{customer['Retention_Action']}</td>
                        </tr>
                    </table>
                </div>
                """
                st.markdown(html_predict, unsafe_allow_html=True)
                
            st.markdown("---")
            
            # Row 2: Flight History Trend Line Chart
            st.markdown("#### **Historical Transaction Activity (2017 - 2018)**")
            
            # Query specific customer activity
            cust_cfa = cfa[cfa['Loyalty Number'] == search_id].sort_values('Date')
            
            if not cust_cfa.empty:
                st.markdown("This timeline maps their monthly bookings (flights taken) and point accumulation trends over the last 24 months. A drop-off in recent months visually demonstrates disengagement.")
                
                fig, ax1 = plt.subplots(figsize=(14, 4.5))
                
                # Primary y-axis (Flights taken)
                color_flights = '#3b82f6' if dark_mode else '#1e3a8a'
                ax1.set_xlabel('Timeline (Month/Year)', fontweight='bold', labelpad=10)
                ax1.set_ylabel('Flights Booked', color=color_flights, fontweight='bold')
                sns.lineplot(data=cust_cfa, x='Date', y='Total Flights', marker='o', color=color_flights, linewidth=2.5, label='Flights Booked', ax=ax1)
                ax1.tick_params(axis='y', labelcolor=color_flights)
                
                # Secondary y-axis (Points accumulated)
                ax2 = ax1.twinx()
                color_points = '#f97316' if dark_mode else '#e67e22'
                ax2.set_ylabel('Points Accumulated', color=color_points, fontweight='bold')
                sns.lineplot(data=cust_cfa, x='Date', y='Points Accumulated', marker='s', color=color_points, linewidth=1.8, linestyle='--', label='Points Accumulated', ax=ax2)
                ax2.tick_params(axis='y', labelcolor=color_points)
                
                # Title and details
                plt.title(f"Customer Loyalty Activity & Engagement Timeline: ID {search_id}", fontsize=14, fontweight='bold', pad=15)
                
                # Combine legends from both axes
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                # Dark Mode Plot Theme overrides
                if dark_mode:
                    ax1.set_facecolor('#1e293b')
                    fig.patch.set_facecolor('#0f172a')
                    
                    # Style grid and tick lines
                    ax1.tick_params(colors='white')
                    ax2.tick_params(colors='white')
                    ax1.xaxis.label.set_color('white')
                    ax1.yaxis.label.set_color(color_flights)
                    ax2.yaxis.label.set_color(color_points)
                    ax1.title.set_color('white')
                    
                    ax1.spines['bottom'].set_color('white')
                    ax1.spines['top'].set_color('none')
                    ax1.spines['left'].set_color('white')
                    ax1.spines['right'].set_color('white')
                    
                    # Style legend
                    legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', facecolor='#1e293b', labelcolor='white')
                    legend.get_title().set_color('white')
                
                fig.tight_layout()
                st.pyplot(fig)
                
                # Additional numeric metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Total Flights Taken (24M)", f"{cust_cfa['Total Flights'].sum():,.0f} flights")
                with col_m2:
                    st.metric("Total Points Accumulated", f"{cust_cfa['Points Accumulated'].sum():,.0f} pts")
                with col_m3:
                    st.metric("Total Points Redeemed", f"{cust_cfa['Points Redeemed'].sum():,.0f} pts")
            else:
                st.warning("No flight transactional history found for this Customer ID in the flight database.")
        else:
            st.error(f"Customer ID `{search_id}` was not found in the loyalty database cohort.")
    except ValueError:
        st.error("Please enter a valid numeric Customer Loyalty ID.")
else:
    st.info("Enter a numeric Customer ID in the search box to view their dashboard profile.")
