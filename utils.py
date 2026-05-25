import streamlit as st
import pandas as pd
import numpy as np
import os

@st.cache_data
def load_data():
    """
    Loads raw flight activity, loyalty history, and the pre-computed machine learning insights.
    Integrates and joins them to form a unified cohort dataset. (Safeguard updated 2026-05-25)
    """
    insights_path = 'final_customer_loyalty_insights.csv'
    clh_path = 'Customer Loyalty History.csv'
    cfa_path = 'Customer Flight Activity.csv'
    
    # Ensure files exist before loading
    if not all(os.path.exists(p) for p in [insights_path, clh_path, cfa_path]):
        st.error("Error: Project CSV files are missing in the workspace directory.")
        st.stop()
        
    insights = pd.read_csv(insights_path)
    clh = pd.read_csv(clh_path)
    cfa = pd.read_csv(cfa_path)
    
    # Standardize dates
    cfa['Date'] = pd.to_datetime(cfa['Year'].astype(str) + '-' + cfa['Month'].astype(str) + '-01')
    
    # Clean salaries
    clh['Salary'] = clh['Salary'].abs()
    median_salary = clh.groupby(['Education', 'Loyalty Card'])['Salary'].transform('median')
    clh['Salary'] = clh['Salary'].fillna(median_salary)
    clh['Salary'] = clh['Salary'].fillna(clh['Salary'].median())
    
    # Merge demographic details with ML insights (Cohort members at 2018-06-01)
    cohort = pd.merge(
        insights,
        clh[['Loyalty Number', 'Gender', 'Education', 'Salary', 'Marital Status', 
             'Loyalty Card', 'Enrollment Year', 'Enrollment Month', 'Province', 'City']],
        on='Loyalty Number',
        how='left'
    )
    
    card_order = {'Star': 1, 'Nova': 2, 'Aurora': 3}
    cohort['Card_Order'] = cohort['Loyalty Card'].map(card_order)
    
    return cohort, cfa

def apply_custom_theme(dark_mode=False):
    """
    Injects custom CSS to style the Streamlit interface with a light or dark theme.
    """
    font_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        .stButton>button {
            background-color: #1e3a8a;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
            border: none;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #1e40af;
            box-shadow: 0 4px 12px rgba(30, 58, 138, 0.25);
        }
    </style>
    """
    st.markdown(font_css, unsafe_allow_html=True)
    
    if dark_mode:
        bg_color = "#0f172a"
        text_color = "#f8fafc"
        card_bg = "#1e293b"
        card_border = "#3b82f6"
        sub_text = "#94a3b8"
        
        style = f"""
        <style>
            .stApp {{
                background-color: {bg_color} !important;
                color: {text_color} !important;
            }}
            [data-testid="stSidebar"] {{
                background-color: {card_bg} !important;
                border-right: 1px solid #334155 !important;
            }}
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
            [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6,
            [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{
                color: {text_color} !important;
            }}
            .metric-card {{
                background-color: {card_bg} !important;
                color: {text_color} !important;
                border-left: 6px solid {card_border} !important;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2) !important;
                padding: 24px;
                border-radius: 14px;
                margin-bottom: 20px;
                transition: transform 0.2s ease-in-out;
            }}
            .metric-card:hover {{
                transform: translateY(-2px);
            }}
            .metric-title {{
                color: {sub_text} !important;
                font-size: 13px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.8px;
                margin-bottom: 6px;
            }}
            .metric-value {{
                color: {text_color} !important;
                font-size: 26px;
                font-weight: 700;
                line-height: 1.1;
            }}
            .metric-subtitle {{
                font-size: 12px;
                color: #34d399 !important;
                font-weight: 600;
                margin-top: 8px;
            }}
            .metric-subtitle-red {{
                font-size: 12px;
                color: #f87171 !important;
                font-weight: 600;
                margin-top: 8px;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: {text_color} !important;
            }}
            .stMarkdown p, .stMarkdown li, .stMarkdown span {{
                color: {text_color} !important;
            }}
            .stAlert {{
                background-color: {card_bg} !important;
                border: 1px solid #334155 !important;
            }}
            .stAlert, .stAlert p, .stAlert span, .stAlert div, .stAlert strong, .stAlert li {{
                color: {text_color} !important;
            }}
            /* Native Streamlit Metrics overrides */
            div[data-testid="stMetricValue"] > div {{
                color: {text_color} !important;
            }}
            div[data-testid="stMetricLabel"] > div {{
                color: {sub_text} !important;
            }}
            /* Native Streamlit Widget Labels */
            label[data-testid="stWidgetLabel"] p {{
                color: {text_color} !important;
            }}
            div[data-testid="stMarkdownContainer"] p {{
                color: {text_color} !important;
            }}
        </style>
        """
    else:
        bg_color = "#f4f6f9"
        text_color = "#0f172a"
        card_bg = "#ffffff"
        card_border = "#1e3a8a"
        sub_text = "#6b7280"
        
        style = f"""
        <style>
            .stApp {{
                background-color: {bg_color} !important;
                color: {text_color} !important;
            }}
            [data-testid="stSidebar"] {{
                background-color: {card_bg} !important;
                border-right: 1px solid #e2e8f0 !important;
            }}
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
            [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6,
            [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{
                color: {text_color} !important;
            }}
            .metric-card {{
                background-color: {card_bg} !important;
                color: {text_color} !important;
                border-left: 6px solid {card_border} !important;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02) !important;
                padding: 24px;
                border-radius: 14px;
                margin-bottom: 20px;
                transition: transform 0.2s ease-in-out;
            }}
            .metric-card:hover {{
                transform: translateY(-2px);
            }}
            .metric-title {{
                color: {sub_text} !important;
                font-size: 13px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.8px;
                margin-bottom: 6px;
            }}
            .metric-value {{
                color: {text_color} !important;
                font-size: 26px;
                font-weight: 700;
                line-height: 1.1;
            }}
            .metric-subtitle {{
                font-size: 12px;
                color: #10b981 !important;
                font-weight: 600;
                margin-top: 8px;
            }}
            .metric-subtitle-red {{
                font-size: 12px;
                color: #ef4444 !important;
                font-weight: 600;
                margin-top: 8px;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: {text_color} !important;
            }}
            /* Native Streamlit Metrics overrides */
            div[data-testid="stMetricValue"] > div {{
                color: {text_color} !important;
            }}
            div[data-testid="stMetricLabel"] > div {{
                color: {sub_text} !important;
            }}
            /* Native Streamlit Widget Labels */
            label[data-testid="stWidgetLabel"] p {{
                color: {text_color} !important;
            }}
            .stAlert {{
                border: 1px solid #e2e8f0 !important;
            }}
            .stAlert, .stAlert p, .stAlert span, .stAlert div, .stAlert strong, .stAlert li {{
                color: #0f172a !important;
            }}
        </style>
        """
        
    st.markdown(style, unsafe_allow_html=True)

def render_kpi_card(title, value, subtitle=None, is_red=False):
    """
    Helper function to render a polished, custom metric card.
    """
    subtitle_class = "metric-subtitle-red" if is_red else "metric-subtitle"
    subtitle_html = f"<div class='{subtitle_class}'>{subtitle}</div>" if subtitle else ""
    
    card_html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {subtitle_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
