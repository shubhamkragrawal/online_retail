"""
Streamlit Frontend for Customer Churn Prediction
User-friendly interface for the FastAPI backend
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json


# extracting metrics from the best model
metrics_details = pd.read_csv('experiment_results.csv')
# print(df.columns)
df_sorted = metrics_details.sort_values(by='f1_score', ascending=False)

# 2. Extract metrics from the top row for columns 'c' and 'd'
top_row = df_sorted.iloc[0]
f1_score = top_row['f1_score']
accuracy = top_row['accuracy']
precision = top_row['precision']
recall = top_row['recall']

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = "http://api:8000"  # Docker service name
# For local development, use: 
# API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .high-risk {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff0000;
    }
    .medium-risk {
        background-color: #fff4cc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffa500;
    }
    .low-risk {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #00ff00;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def predict_churn(customer_data):
    """Send prediction request to API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def create_gauge_chart(probability):
    """Create gauge chart for churn probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_feature_importance_chart(features_dict):
    """Create bar chart of input features"""
    # Select top features to display
    display_features = {
        'Monetary': features_dict.get('Monetary', 0),
        'Frequency': features_dict.get('Frequency', 0),
        'Recency': features_dict.get('Recency', 0),
        'Avg Order Value': features_dict.get('TotalPrice_mean', 0),
        'Product Variety': features_dict.get('StockCode_nunique', 0),
        'Customer Lifetime': features_dict.get('CustomerLifetime', 0)
    }
    
    df = pd.DataFrame(list(display_features.items()), columns=['Feature', 'Value'])
    
    fig = px.bar(
        df,
        x='Value',
        y='Feature',
        orientation='h',
        title='Customer Profile',
        color='Value',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<p class="main-header">🛒 Customer Churn Prediction System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        
        st.title("📊 Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Home", "🔮 Single Prediction", "ℹ️ About"]
        )
        
        st.markdown("---")
        
        # API Status
        st.subheader("API Status")
        if check_api_health():
            st.success("✅ API Connected")
            model_info = get_model_info()
            if model_info:
                st.info(f"**Model:** {model_info['model_name']}")
                st.info(f"**Features:** {model_info['features_count']}")
                st.info(f"**PCA:** {'Yes' if model_info['uses_pca'] else 'No'}")
        else:
            st.error("❌ API Unavailable")
            st.warning("Please ensure the API service is running")
    
    # Main content
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Single Prediction":
        show_prediction_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Display home page"""
    st.header("Welcome to Customer Churn Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    
    with col1:
         st.markdown("""
        <div class="metric-card">
            <h3 style="color: black;">🎯 Accurate Predictions</h3>
            <p style="color: black;">Machine learning model trained on 16 experiments with F1-score optimization</p>
        </div>
    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: black;">⚡ Real-time Analysis</h3>
            <p style="color: black;">Get instant churn predictions through our FastAPI backend</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: black;">📈 Actionable Insights</h3>
            <p style="color: black;">Identify high-risk customers and take preventive action</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🚀 Quick Start")
    st.markdown("""
    1. **Single Prediction**: Enter customer details manually for instant prediction
    2. **Risk Levels**: 
        - 🟢 Low Risk (< 30%)
        - 🟡 Medium Risk (30-70%)
        - 🔴 High Risk (> 70%)
    """)
    
    st.subheader("📊 Model Performance")
   
    # Mock performance metrics (replace with actual from experiments)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("F1-Score", f"{f1_score:.2f}", "±0.03")
    col2.metric("Accuracy", f"{accuracy:.2f}", "±0.02")
    col3.metric("Precision", f"{precision:.2f}", "±0.04")
    col4.metric("Recall", f"{recall:.2f}", "±0.03")

def show_prediction_page():
    """Display single prediction page"""
    st.header("🔮 Single Customer Prediction")
    
    st.markdown("Enter customer information to predict churn probability")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("RFM Metrics")
            recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30, help="Number of days since the customer's last purchase")
            frequency = st.number_input("Frequency (number of orders)", min_value=0, value=5, help="Total number of orders placed")
            monetary = st.number_input("Monetary (total spent £)", min_value=0.0, value=500.0, help="Total amount spent by customer")
        
        with col2:
            st.subheader("Purchase Behavior")
            unique_invoices = st.number_input("Unique Invoices", min_value=0, value=5)
            quantity_sum = st.number_input("Total Quantity", min_value=0.0, value=50.0)
            quantity_mean = st.number_input("Avg Quantity/Order", min_value=0.0, value=10.0)
            total_price_mean = st.number_input("Avg Order Value £", min_value=0.0, value=100.0)
        
        with col3:
            st.subheader("Customer Profile")
            unique_products = st.number_input("Unique Products", min_value=0, value=10)
            customer_lifetime = st.number_input("Customer Lifetime (days)", min_value=0, value=180)
            avg_days_between = st.number_input("Avg Days Between Purchases", min_value=0.0, value=30.0)
            
            # Country selection
            country = st.selectbox("Country", [
                "United Kingdom", "Germany", "France", "Ireland", "Spain",
                "Netherlands", "Belgium", "Switzerland", "Portugal", "Australia", "Other"
            ])
        
        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
    
    if submitted:
        # Prepare customer data
        customer_data = {
            "Recency": float(recency),
            "Frequency": int(frequency),
            "Monetary": float(monetary),
            "InvoiceNo_nunique": int(unique_invoices),
            "Quantity_sum": float(quantity_sum),
            "Quantity_mean": float(quantity_mean),
            "TotalPrice_sum": float(monetary),
            "TotalPrice_mean": float(total_price_mean),
            "TotalPrice_std": 0.0,
            "StockCode_nunique": int(unique_products),
            "CustomerLifetime": float(customer_lifetime),
            "AvgDaysBetweenPurchases": float(avg_days_between),
        }
        
        # Add country one-hot encoding
        countries = ["United_Kingdom", "Germany", "France", "EIRE", "Spain",
                    "Netherlands", "Belgium", "Switzerland", "Portugal", "Australia"]
        for c in countries:
            customer_data[f"Country_{c}"] = 1 if country.replace(" ", "_") == c else 0
        
        # Make prediction
        with st.spinner("Analyzing customer..."):
            result = predict_churn(customer_data)
        
        if result:
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Display risk level with color
            risk_level = result['risk_level']
            risk_class = f"{risk_level.lower()}-risk"
            
            st.markdown(f"""
            <div class="{risk_class}">
                <h2>Risk Level: {risk_level}</h2>
                <p style="font-size: 1.2rem;">Churn Probability: {result['churn_probability']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create two columns for visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Gauge chart
                fig_gauge = create_gauge_chart(result['churn_probability'])
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # Feature chart
                fig_features = create_feature_importance_chart(customer_data)
                st.plotly_chart(fig_features, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if risk_level == "High":
                st.error("""
                **Immediate Action Required:**
                - Reach out with personalized retention offer
                - Offer exclusive discount or loyalty rewards
                - Schedule customer service call
                - Provide VIP support
                """)
            elif risk_level == "Medium":
                st.warning("""
                **Monitor and Engage:**
                - Send targeted email campaigns
                - Offer product recommendations
                - Provide special promotions
                - Monitor purchase behavior closely
                """)
            else:
                st.success("""
                **Maintain Satisfaction:**
                - Continue excellent service
                - Request feedback and reviews
                - Offer loyalty program benefits
                - Send new product updates
                """)

def show_about_page():
    """Display about page"""
    st.header("ℹ️ About This System")
    
    st.markdown(f"""
    ### 🎯 Project Overview
    This Customer Churn Prediction System uses machine learning to identify customers
    at risk of churning in an online retail environment.
    
    ### 🔬 Technical Details
    - **Dataset**: UCI Online Retail Dataset
    - **Problem Type**: Binary Classification (Churn vs. Retain)
    - **Database**: SQLite with 3NF normalization
    - **Models Tested**: 16 experiments across 4 algorithms
      - Logistic Regression
      - Random Forest
      - XGBoost
      - Support Vector Machine (SVM)
    - **Preprocessing**: StandardScaler + Optional PCA
    - **Hyperparameter Tuning**: Optuna (50 trials per model)
    - **Experiment Tracking**: MLflow/DagsHub
    
    ### 📊 Features Used
    - **RFM Analysis**: Recency, Frequency, Monetary
    - **Purchase Behavior**: Order patterns, quantity metrics
    - **Customer Profile**: Lifetime, product variety, country
    
    ### 🏗️ Architecture
    - **Frontend**: Streamlit
    - **Backend**: FastAPI
    - **Deployment**: Docker + Docker Compose
    - **Database**: SQLite (3NF normalized)
    
    ### 👨‍💻 Development
    - **ML Pipeline**: Scikit-learn, XGBoost
    - **Experiment Tracking**: MLflow + DagsHub
    - **Containerization**: Docker
    - **API**: FastAPI with Pydantic validation
    
    ### 📈 Model Performance
    The best performing model achieved:
    - F1-Score: ~{f1_score:.2f}
    - Accuracy: ~{accuracy:.2f}
    - Precision: ~{precision:.2f}
    - Recall: ~{recall:.2f}

    ### 📝 License
    This project is for educational purposes.
    """)

if __name__ == "__main__":
    main()