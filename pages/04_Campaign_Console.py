import streamlit as st
import pandas as pd
import numpy as np
import time
import utils

# Set page config
st.set_page_config(
    page_title="Campaign Console - FlyCaelum",
    page_icon="✈️",
    layout="wide"
)

# Apply styling
dark_mode = st.session_state.get('dark_mode', False)
utils.apply_custom_theme(dark_mode)

# Load filtered cohort (with outdated cache safeguard)
if 'filtered_df' in st.session_state and 'BDS_Score' in st.session_state['filtered_df'].columns:
    filtered_df = st.session_state['filtered_df']
else:
    cohort, cfa = utils.load_data()
    st.session_state['filtered_df'] = cohort
    st.session_state['cfa'] = cfa
    filtered_df = cohort

st.title("🎯 Campaign Trigger Console")
st.subheader("Simulate targeted loyalty campaigns and calculate financial return on investment (ROI)")
st.markdown("---")

# Row 1: Campaign Selector and Details
st.markdown("### ⚡ Step 1: Select Retention Campaign")

campaigns = {
    "Priority Reactivation Concierge": {
        "persona": "At-Risk Elites",
        "description": "High-touch phone outreach and a custom status-extension voucher for former elite members who have stopped flying.",
        "default_offer_cost": 100
    },
    "Welcome Miles Bonus & Discount": {
        "persona": "New Members (Onboarding)",
        "description": "Welcome bonus miles and a discount voucher to convert new signups into active fliers.",
        "default_offer_cost": 30
    },
    "VIP Lounge Access & Perks": {
        "persona": "Loyal Champions",
        "description": "Appreciation perks, lounge passes, and early flight booking access to retain our top-value fliers.",
        "default_offer_cost": 50
    },
    "Double Miles & Cabin Upgrades": {
        "persona": "Corporate Flyers",
        "description": "Premium seat upgrade opportunities and double miles on their favorite business routes.",
        "default_offer_cost": 150
    },
    "Discounted Vacation Packages": {
        "persona": "Active Leisure",
        "description": "Direct vacation discounts, point redemption promos, and travel partner deals.",
        "default_offer_cost": 75
    }
}

selected_campaign = st.selectbox("Choose a Targeted Campaign Playbook", options=list(campaigns.keys()))

c_info = campaigns[selected_campaign]
target_persona = c_info['persona']
segment_df = filtered_df[filtered_df['Persona'] == target_persona]

# Calculate target segment metrics
segment_size = len(segment_df)
segment_avg_clv = segment_df['CLV'].mean() if segment_size > 0 else 0
segment_total_clv = segment_df['CLV'].sum() if segment_size > 0 else 0

col_det1, col_det2, col_det3 = st.columns(3)

with col_det1:
    st.metric("Target Segment Size", f"{segment_size:,} members", help="Active members matching this persona")
with col_det2:
    st.metric("Average Customer Value (CLV)", f"${segment_avg_clv:,.2f} CAD", help="Average lifetime bookings per customer")
with col_det3:
    st.metric("Total Portfolio Value at Risk", f"${segment_total_clv:,.2f} CAD", help="Sum of CLV for all members in this segment")

st.markdown(f"**Campaign Description**: {c_info['description']}")


st.markdown("---")

# Row 2: Interactive ROI Controls
st.markdown("### 🎛️ Step 2: Configure ROI Simulator Parameters")
st.markdown("Adjust these sliders to simulate different campaign costs, response rates, and financial returns.")

col_control1, col_control2 = st.columns(2)

with col_control1:
    success_rate = st.slider(
        "Expected Campaign Success Rate (%)",
        min_value=1.0,
        max_value=50.0,
        value=15.0,
        step=1.0,
        help="The percentage of target members who respond to the offer and resume flight activity."
    ) / 100.0

with col_control2:
    offer_cost = st.slider(
        "Average Campaign Cost per Member (CAD)",
        min_value=10,
        max_value=300,
        value=c_info['default_offer_cost'],
        step=5,
        help="The marketing, voucher, or loyalty point expense allocated per targeted customer."
    )

# Ghost Member Suppression Safeguard
ghosts = filtered_df[filtered_df['Persona'] == "Ghost Members"]
num_ghosts = len(ghosts)
st.warning(f"🚫 **Operational Safeguard**: {num_ghosts} Ghost Members (accounts with tenure > 6 months, zero flights, and zero points history) are automatically excluded from target marketing lists. Under our business rules, this suppression prevents list pollution and immediately saves ${num_ghosts * offer_cost:,.2f} CAD in potential marketing waste (calculated at the current per-member cost of {offer_cost:.2f} CAD).")

st.markdown("---")

# Row 3: Run Simulation
st.markdown("### 🚀 Step 3: Run Campaign Simulation & Calculate ROI")

if st.button("Run Simulation Campaign"):
    st.markdown("#### **Simulation Outcome**")
    
    with st.spinner("Analyzing cohort response, applying propensity scores, and calculating ROI..."):
        time.sleep(1.2) # simulated latency
        
    # Math calculations
    total_campaign_cost = segment_size * offer_cost
    reactivated_members = int(segment_size * success_rate)
    gross_value_saved = reactivated_members * segment_avg_clv
    net_savings = gross_value_saved - total_campaign_cost
    roi = (net_savings / total_campaign_cost) * 100.0 if total_campaign_cost > 0 else 0.0
    
    # Show outcome metrics
    col_out1, col_out2, col_out3, col_out4 = st.columns(4)
    
    with col_out1:
        utils.render_kpi_card(
            title="Total Campaign Cost",
            value=f"${total_campaign_cost:,.2f} CAD",
            subtitle=f"${offer_cost} allocated per member"
        )
        
    with col_out2:
        utils.render_kpi_card(
            title="Reactivated Customers",
            value=f"{reactivated_members:,} members",
            subtitle=f"{success_rate:.0%} response rate simulated"
        )
        
    with col_out3:
        utils.render_kpi_card(
            title="Gross Revenue Saved",
            value=f"${gross_value_saved:,.2f} CAD",
            subtitle="Lifetime booking value retained"
        )
        
    with col_out4:
        is_negative = net_savings < 0
        sign = "-" if is_negative else "+"
        utils.render_kpi_card(
            title="Net Business Value",
            value=f"${net_savings:,.2f} CAD",
            subtitle=f"Estimated ROI: {sign}{abs(roi):,.1f}%",
            is_red=is_negative
        )
        
    # Narrative justification
    if net_savings > 0:
        st.success(f"🎉 **Positive Return on Investment**: The campaign generates a net saved value of **${net_savings:,.2f} CAD** after accounting for the marketing budget. This represents a highly viable retention initiative.")
    else:
        st.error(f"⚠️ **Negative Return on Investment**: Under these parameters, the marketing spend exceeds the saved lifetime value by **${abs(net_savings):,.2f} CAD**. We recommend reducing the offer cost or improving the target criteria to raise the response rate.")
else:
    st.info("Click the 'Run Simulation Campaign' button to view the simulated cost, reactivated members, gross saved revenue, and net ROI.")
