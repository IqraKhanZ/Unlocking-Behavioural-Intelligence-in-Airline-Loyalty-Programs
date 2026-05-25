import streamlit as st
import utils

# Set page config (MUST be the first Streamlit call)
st.set_page_config(
    page_title="FlyCaelum Loyalty Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize and persist Dark Mode state in session state
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

# Sidebar logo and Interface Settings
st.sidebar.image("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?q=80&w=400&auto=format&fit=crop", caption="FlyCaelum Behavioral Intelligence", use_container_width=True)
st.sidebar.markdown("### Settings")
dark_mode = st.sidebar.toggle("Dark Mode Toggle", value=st.session_state['dark_mode'])
st.session_state['dark_mode'] = dark_mode

# Apply selected theme variables
utils.apply_custom_theme(dark_mode)

# Define page view wrapper for home
def show_home():
    import home
    home.render_page()

# Setup programmatic multi-page navigation (renaming default 'app' to 'Home')
home_page = st.Page(show_home, title="Home", icon="🏠", default=True)
churn_page = st.Page("pages/01_Churn_Monitoring.py", title="Churn Risk Monitoring", icon="🔍")
segment_page = st.Page("pages/02_Customer_Segmentation.py", title="Customer Segmentation", icon="👥")
drilldown_page = st.Page("pages/03_Customer_Drilldown.py", title="Customer Drilldown", icon="🔎")
campaign_page = st.Page("pages/04_Campaign_Console.py", title="Campaign Console", icon="🎯")
survival_page = st.Page("pages/05_Survival_Analysis.py", title="Survival Analysis", icon="⏳")

# Set up navigation structure
pg = st.navigation({
    "Main Views": [home_page, churn_page, segment_page, drilldown_page, campaign_page, survival_page]
})

# Run the selected page
pg.run()

