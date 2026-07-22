import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Fraud Detection Prediction System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fraud Detection SVM Prediction App")
st.write("This version automatically extracts feature names from your model/scaler and auto-aligns your dataset to exactly 22 columns.")

@st.cache_resource
def load_model_and_scaler():
    with open("svm_model.pkl", "rb") as file:
        model = pickle.load(file)
    try:
        with open("scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
    except FileNotFoundError:
        scaler = None
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
    st.success("✅ SVM Model and Scaler successfully loaded!")
    
    EXPECTED_FEATURES = 22

    # Attempt to automatically find the exact 22 column names used during training
    trained_feature_names = None
    if scaler is not hasattr(scaler, 'feature_names_in_') and hasattr(model, 'feature_names_in_'):
        trained_feature_names = model.feature_names_in_
    elif scaler is not None and hasattr(scaler, 'feature_names_in_'):
        trained_feature_names = scaler.feature_names_in_

    uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("### Raw Upload Preview:")
        st.dataframe(df.head(5))
        
        # 1. Separate base numeric columns
        numeric_features = [
            'account_age_days', 'avg_monthly_spend', 'merchant_risk_score', 
            'transaction_amount', 'is_international', 'ip_risk_score', 
            'txn_count_1h', 'txn_count_24h', 'failed_txn_count_24h', 
            'geo_distance_from_last_txn', 'amount_deviation_from_user_mean', 
            'post_auth_risk_score'
        ]
        existing_numeric = [col for col in numeric_features if col in df.columns]
        base_df = df[existing_numeric].copy()
        
        # 2. Extract and force categorical features to string format so get_dummies actually expands them
        categorical_features = ['credit_score_band', 'kyc_level']
        existing_categorical = [col for col in categorical_features if col in df.columns]
        
        if existing_categorical:
            # Crucial Fix: Convert to string type so pandas treats them as text categories
            cat_df = df[existing_categorical].astype(str)
            dummy_df = pd.get_dummies(cat_df, dtype=float)
            processed_df = pd.concat([base_df, dummy_df], axis=1)
        else:
            processed_df = base_df.copy()
            
        processed_df = processed_df.astype(float)
        
        # 3. SMART COLUMN ALIGNMENT LAYER
        if trained_feature_names is not None:
            # If we successfully found the 22 training feature names, reindex the columns perfectly!
            # Missing columns are automatically filled with 0.0, and extra columns are dropped.
            final_df = processed_df.reindex(columns=trained_feature_names, fill_value=0.0)
        else:
            # Fallback: if names aren't embedded inside the pickle files, adjust shape directly
            final_df = processed_df.copy()
            current_cols = final_df.shape[1]
            if current_cols < EXPECTED_FEATURES:
                # Pad with 0s until it reaches 22 columns
                for i in range(EXPECTED_FEATURES - current_cols):
                    final_df[f'padded_feature_{i}'] = 0.0
            elif current_cols > EXPECTED_FEATURES:
                # Take only the first 22 columns
                final_df = final_df.iloc[:, :EXPECTED_FEATURES]

        # Ensure the matrix passed is exactly 22 columns
        test_data = final_df.values
        
        if test_data.shape[1] == EXPECTED_FEATURES:
            # Apply your saved 22-feature StandardScaler
            if scaler is not None:
                test_data = scaler.transform(test_data)
                
            # Perform predictions
            predictions = model.predict(test_data)
            
            # Show the final results screen
            df_results = df.copy()
            df_results.insert(0, '🎯 Fraud_Prediction', predictions)
            
            st.markdown("### 🎯 Prediction Results:")
            st.success(f"Successfully evaluated {len(predictions):,} rows!")
            st.dataframe(df_results.head(1000))
            
            col_a, col_b = st.columns(2)
            col_a.metric("Legitimate Transactions (Class 0)", f"{(predictions == 0).sum():,}")
            col_b.metric("Fraudulent Transactions (Class 1)", f"{(predictions == 1).sum():,}")
        else:
            st.error(f"⛔ Core Error: Shape alignment failed. Matrix has {test_data.shape[1]} columns instead of {EXPECTED_FEATURES}.")

except FileNotFoundError:
    st.error("❌ Critical Error: Make sure 'svm_model.pkl' and 'scaler.pkl' are placed in the exact same directory.")
except Exception as e:
    st.error(f"💥 Application Error: {e}")