import os
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. PAGE CONFIGURATION & KAGGLE SETUP
# ==========================================
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="🛒",
    layout="wide"
)

# Set Kaggle credentials directly in environment variables
os.environ['KAGGLE_USERNAME'] = "ayup"
os.environ['KAGGLE_KEY'] = "KGAT_af5d4ee16d2d9de8e9eae1e7fc43bab8"


# ==========================================
# 2. DATA LOADING & UNZIPPING
# ==========================================
@st.cache_data
def load_dataset():
    """Downloads and extracts the dataset if not present locally."""
    if not os.path.exists("train.csv"):
        # Option A: Unzip local archive if present
        if os.path.exists("walmart-sales-forecast.zip"):
            with zipfile.ZipFile("walmart-sales-forecast.zip", 'r') as zip_ref:
                zip_ref.extractall()
        # Option B: Download directly using Kaggle API
        else:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files('aslanahmedov/walmart-sales-forecast', unzip=True)

    df = pd.read_csv('train.csv')
    return df


# ==========================================
# 3. SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("⚙️ Model Parameters")

n_estimators = st.sidebar.slider(
    "Number of Trees (n_estimators)", 
    min_value=5, 
    max_value=100, 
    value=10, 
    step=5
)

test_size = st.sidebar.slider(
    "Test Split Ratio", 
    min_value=0.1, 
    max_value=0.4, 
    value=0.3, 
    step=0.05
)

random_state = st.sidebar.number_input(
    "Random State", 
    value=42, 
    step=1
)


# ==========================================
# 4. MAIN APP INTERFACE
# ==========================================
st.title("🛒 Walmart Weekly Sales Prediction App")
st.markdown("Automated dataset retrieval, Random Forest training, and sales prediction.")

st.divider()

# Load Data
try:
    with st.spinner("Loading dataset..."):
        df = load_dataset()
    st.success("Data successfully loaded!")

    # Tabs for Organization
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🤖 Model Training", "🔮 Predict Sales"])

    # --------------------------------------
    # TAB 1: DATA OVERVIEW
    # --------------------------------------
    with tab1:
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Summary Statistics")
            st.dataframe(df.describe().T)
        with col2:
            st.subheader("Missing Values Count")
            st.dataframe(df.isnull().sum().to_frame(name="Null Count"))

    # --------------------------------------
    # TAB 2: MODEL TRAINING
    # --------------------------------------
    with tab2:
        st.subheader("Train Random Forest Regressor")
        
        if st.button("🚀 Start Training Pipeline", type="primary"):
            with st.spinner("Processing features and fitting model..."):
                # Feature Preparation
                X = df.drop('Weekly_Sales', axis=1)
                y = df['Weekly_Sales']
                
                # One-Hot Encoding
                X_encoded = pd.get_dummies(X)
                
                # Train/Test Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y, test_size=test_size, random_state=random_state
                )
                
                # Feature Scaling
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Model Fitting
                model = RandomForestRegressor(
                    n_estimators=n_estimators, 
                    random_state=random_state
                )
                model.fit(X_train_scaled, y_train)
                
                # Model Evaluation
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Store trained artifacts in session state
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['columns'] = X_encoded.columns.tolist()

            st.success("Model Training Complete!")
            
            # Display Metric
            st.metric(
                label="Root Mean Squared Error (RMSE)", 
                value=f"${rmse:,.2f}"
            )

    # --------------------------------------
    # TAB 3: CUSTOM PREDICTIONS
    # --------------------------------------
    with tab3:
        st.subheader("Predict Weekly Sales for Custom Inputs")
        
        if 'model' not in st.session_state:
            st.warning("Please train the model under the **Model Training** tab first.")
        else:
            col_input1, col_input2, col_input3 = st.columns(3)
            
            with col_input1:
                store = st.number_input("Store ID", min_value=int(df['Store'].min()), max_value=int(df['Store'].max()), value=1)
            with col_input2:
                dept = st.number_input("Department ID", min_value=int(df['Dept'].min()), max_value=int(df['Dept'].max()), value=1)
            with col_input3:
                is_holiday = st.selectbox("Is Holiday Week?", [False, True])
                
            input_date = st.date_input("Date")
            
            if st.button("Predict Weekly Sales"):
                # Construct input dataframe
                input_df = pd.DataFrame([{
                    'Store': store,
                    'Dept': dept,
                    'Date': str(input_date),
                    'IsHoliday': is_holiday
                }])
                
                # Align columns with dummy encoding used during training
                input_encoded = pd.get_dummies(input_df)
                input_encoded = input_encoded.reindex(columns=st.session_state['columns'], fill_value=0)
                
                # Scale input
                input_scaled = st.session_state['scaler'].transform(input_encoded)
                
                # Predict
                prediction = st.session_state['model'].predict(input_scaled)[0]
                
                st.metric(
                    label="Estimated Weekly Sales", 
                    value=f"${prediction:,.2f}"
                )

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")