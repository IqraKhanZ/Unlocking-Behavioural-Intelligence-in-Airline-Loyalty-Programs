# FlyCaelum Airline Loyalty Behavioral Intelligence Dashboard

## 🚀 Deployment

The live interactive dashboard is deployed and available at the link below:

🔗 **[View Live Dashboard](https://unlocking-behavioural-intelligence-in-airline-loyalty-programs.streamlit.app/Churn_Monitoring)**

Welcome to the **FlyCaelum Loyalty Behavioral Intelligence Dashboard**. This application is a professional, production-grade business tool designed to translate predictive machine learning outputs into high-impact customer retention strategies. 

By targeting **"silent churn"** (loyalty members who stop flying without explicitly cancelling their accounts) rather than reactive retention, this platform helps FlyCaelum Airlines identify revenue risk, segment customers, and optimize marketing spend for targeted retention campaigns.

---

## 📋 Table of Contents
1. [Project & Business Context](#-project--business-context)
2. [Architecture Design](#-architecture-design)
3. [Page-by-Page Feature Tour](#-page-by-page-feature-tour)
4. [Light & Dark Mode Visual Styling](#-light--dark-mode-visual-styling)
5. [Installation & Setup](#-installation--setup)
6. [Data Dictionary & Key Metrics](#-data-dictionary--key-metrics)

---

## ✈️ Project & Business Context

Airlines face high customer acquisition costs. Customer loyalty programs are designed to drive repeat business, but many programs fail to identify customer disengagement until it is too late. 

### **The Business Objective**
- **Proactive Churn Detection**: Identify customers at risk of disengaging (zero flight activity in the subsequent 6 months) using historical monthly flight transactions and demographics.
- **Value-at-Risk Quantization**: Connect predicted churn probability to Customer Lifetime Value (CLV) to prioritize high-value savers.
- **Trigger-Based Marketing**: Map specific customer personas to customized retention plays with real-time campaign ROI simulations.
- **Behavioral Disengagement Scoring**: Implement the **Behavioral Disengagement Score (BDS)** as a weighted composite score to define **Soft Churn** (BDS > 0.55, behaviorally inactive but not formally cancelled) vs **Hard Churn** (formally cancelled).

---

## 🏗️ Architecture Design

To ensure optimal performance and separate data science workloads from application delivery, this project uses a **decoupled, model-output-driven architecture**:

 ```
  ┌────────────────────────────────────────────────────────────────────┐      ┌──────────────────────────────────┐
  │Unlocking Behavioural Intelligence in Airline Loyalty Programs.ipynb│ ───> │ final_customer_loyalty_insights  │
  │                      (ML Pipeline & Modeling)                      │      │   (Pre-calculated predictions)   │
  └────────────────────────────────────────────────────────────────────┘      └──────────────────────────────────┘
                                                     │
                                                     ▼
  ┌────────────────────────────────────────────────────────────────────┐      ┌──────────────────────────────────┐
  │                         utils.py / app.py                          │ <─── │     Streamlit Multi-page App     │
  │                      (Data loader & Styling)                       │      │      (Dynamic Interactive UI)    │
  └────────────────────────────────────────────────────────────────────┘      └──────────────────────────────────┘
 ```

1. **Modeling & Feature Engineering (`Unlocking Behavioural Intelligence in Airline Loyalty Programs.ipynb`)**:
   - Cleans the raw data (resolves negative salaries, imputes missing records).
   - Prevents target leakage using a **temporal split snapshot cohort** split (t0 = 2018-06-01 snapshot).
   - Defines the behavioral churn target variable using the **Behavioral Disengagement Score (BDS)**.
   - Trains XGBoost and Random Forest classifiers (achieving temporal test cohort ROC-AUC ≈ 0.942 and PR-AUC ≈ 0.867).
   - Clusters members into 5 distinct behavioral personas using KMeans, overlaid with rules to generate **6 customer personas** (Loyal Champions, Corporate Flyers, Active Leisure, New Members (Onboarding), Ghost Members, and At-Risk Elites).
   - Performs **Survival Analysis** (Kaplan-Meier survival curves and Cox Proportional Hazards regression) to calculate retention milestones and hazard ratios of tenure.
   - Exports the final predictions, risk scores, BDS scores, churn classifications, and segment mappings to a static CSV database (`final_customer_loyalty_insights.csv`).

2. **Streamlit Interactive UI (`app.py` & pages)**:
   - Programmatically reads the pre-calculated insights from the CSV.
   - Avoids retraining models on every page load, ensuring lightning-fast load times.
   - Implements **Outdated Cache Safeguards** on all subpages to automatically clear stale session data and reload the fresh 12-column insights database.

---

## 🚀 Page-by-Page Feature Tour

The dashboard is structured into 6 logical views, which can be selected via the sidebar navigation menu:

### **1. Executive Home (`home.py`)**
- **Purpose**: The main landing page providing high-level KPIs and business context for executives.
- **Features**:
  - **KPI Metric Grid**: Displays active loyalty members, average portfolio churn risk, total portfolio Value-at-Risk (sum of CLV for members with >50% churn probability), and estimated campaign savings.
  - **Global Filtering Panel**: Interactive sidebar filters (Province, Loyalty Card Level, Marital Status, Gender) that dynamically update metrics across all sub-pages.
  - **Customer Persona Distribution**: A horizontal bar chart showing the composition of the active cohort (including Ghost Members).
  - **Dynamic Highlights**: Reactivation savings calculations are computed dynamically based on the current cohort data.

### **2. Churn Risk Monitoring (`pages/01_Churn_Monitoring.py`)**
- **Purpose**: Dive deep into the machine learning predictions and identify specific high-risk customers.
- **Features**:
  - **Churn Risk Probability Distribution**: Histogram illustrating the count of members across predicted risk brackets.
  - **Model Risk Drivers**: Horizontal feature importance chart showing XGBoost's key decision factors (headed by `Points_Velocity` at 38.6% and `Redemption_Ratio` at 30.2%).
  - **High-Risk Member Registry**: An interactive registry table showing member IDs, card tiers, CLV, predicted risk, **BDS Score**, **Churn Classification** (Soft Churn vs Hard Churn), and recommended retention campaigns.

### **3. Customer Segmentation (`pages/02_Customer_Segmentation.py`)**
- **Purpose**: Inspect customer personas and understand demographic breakdowns.
- **Features**:
  - **Value-vs-Risk Scatter Matrix**: A 2D plot mapping CLV against Churn Probability, colored by the 6 customer personas.
  - **Segment Profile Matrix**: A summary table displaying averages for CLV, annual salary, flight frequency, and points history for each persona (including Ghost Members).
  - **CLV Comparison Table**: Compares CLV averages between active ($7,957.22) and disengaged ($8,039.21) cohorts, illustrating that value history is not a churn shield.
  - **Demographic Breakdown Grid**: A 4-panel chart breakdown (Loyalty Card tier, Education level, Marital Status, and Salary distribution) for the selected persona.

### **4. Customer Drilldown (`pages/03_Customer_Drilldown.py`)**
- **Purpose**: Empower account managers to research individual customer records.
- **Features**:
  - **Search Bar Lookup**: Enter any Loyalty ID (samples are provided for ease of testing).
  - **Demographic & Value Cards**: Displays customized demographic and ML insights cards side-by-side (including **BDS Score** and **Churn Classification** lookup fields).
  - **Historical Transaction Timeline**: Renders a dual-axis line chart mapping monthly flights taken and points accumulated over the last 24 months, visualizing exactly when "silent churn" began.

### **5. Campaign Console (`pages/04_Campaign_Console.py`)**
- **Purpose**: Simulate retention campaigns and calculate ROI before committing marketing budget.
- **Features**:
  - **Target Segment Selection**: Select a campaign playbook (e.g., *Priority Reactivation Concierge*, *VIP Lounge Access*) and view the target segment size and total value at risk.
  - **Campaign Block Rules**: Ghost Members are automatically excluded from campaigns to prevent marketing waste.
  - **ROI Parameters Control**: Adjust sliders for expected campaign response rate and per-member cost.
  - **ROI Outcome Simulation**: Run the simulation to instantly compute total campaign cost, reactivated members, gross saved revenue, net business value saved, and return on investment (ROI %).

### **6. Survival Analysis (`pages/05_Survival_Analysis.py`)**
- **Purpose**: Fit and visualize survival functions to understand the timing of customer churn and identify risk covariates.
- **Features**:
  - **Kaplan-Meier Survival Curves**: Plots the probability of remaining engaged over time, stratified by loyalty card tier (Star, Nova, Aurora), showing a major drop at 72 months.
  - **Stratified Retention Milestones Table**: Displays the statistical retention percentages at key milestones (12, 24, 36, 48, 60, and 72 months).
  - **Cox Proportional Hazards Summary**: Renders live regression statistics (coefficients, hazard ratios, standard errors, and z-scores) for loyalty program tenure, flight recency, and customer lifetime value.

---

## 🎨 Light & Dark Mode Visual Styling

The dashboard features a **dynamic theme toggle** located in the sidebar settings panel. 

- **Aviation Aesthetics**: Custom CSS uses the **Outfit** Google Font, smooth drop shadows, clean borders, and premium brand colors.
- **Dynamic Chart Adjustments**: Custom Matplotlib and Seaborn plots dynamically adapt to dark mode (updating background colors, axis lines, and tick labels to white) so the charts match the layout seamlessly.
- **Component Styling**: Metric cards and native Streamlit input widgets are styled dynamically to ensure high contrast and readability under both light and dark settings.

---

## ⚙️ Installation & Setup

### **Prerequisites**
Make sure you have Python 3.8+ installed on your system.

### **1. Clone or Move to the Project Directory**
Ensure you are in the workspace folder:
```bash
cd "C:\Users\iqraf\Downloads\IIT Summer Project"
```

### **2. Install Dependencies**
Install all required packages using the project `requirements.txt` file (including `lifelines` for survival analysis, `xgboost` for ML prediction, and `xhtml2pdf` for PDF generation):
```bash
pip install -r requirements.txt
```

### **3. Run the Streamlit Application**
Start the Streamlit development server:
```bash
python -m streamlit run app.py
```
Once launched, the app will automatically open in your default browser at: **`http://localhost:8501`**

---


## 📊 Data Dictionary & Key Metrics

Here are the key metrics and features used inside the dashboard:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| **Loyalty Number** | Integer | Unique identifier for the customer. |
| **CLV (Customer Lifetime Value)** | Float | Estimated booking value of the customer in CAD. |
| **Churn Probability** | Float | XGBoost model output representing the likelihood of disengagement. |
| **BDS_Score** | Float | Composite score measuring flight inactivity (40%), redemption dormancy (30%), and point momentum (30%). |
| **Churn_Target** | Binary | Final target label (1 if BDS > 0.55 or cancelled, 0 otherwise). |
| **Churn_Type** | Categorical | Churn classification of the customer (Active, Soft Churn, Hard Churn). |
| **Tenure_Months** | Integer | Months active in the loyalty program. |
| **Recency_Flights** | Integer | Months since the customer's last booked flight. |
| **Flights_Last_12M** | Float | Average flights taken in the last 12 months. |
| **Net_Points_History** | Float | Total accumulated loyalty points minus points redeemed. |
| **Persona** | Categorical | Customer group clustered via KMeans and overrides (Champions, Corporate, Leisure, Onboarding, Ghost Members, At-Risk Elites). |
| **Retention_Action** | Categorical | Best recommended campaign playbook based on customer segment. |

---
*Developed by Iqra Khan*
