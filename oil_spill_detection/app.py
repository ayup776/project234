import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ==========================================
# 1. ADVANCED PAGE CONFIGURATION & THEMING
# ==========================================
st.set_page_config(
    page_title="DeepSpill AI | Marine Oil Spill Detection",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, enterprise-grade dark/modern UI
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            font-family: 'Inter', sans-serif;
        }
        div[data-testid="stMetricSimpleValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #00f2fe !important;
        }
        .css-1d391kg {
            background-color: #161b22 !important;
        }
        .stAlert {
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DEEP LEARNING MODEL ARCHITECTURE
# ==========================================
def build_cnn_architecture():
    """Reconstructs the exact CNN architecture layout."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

@st.cache_resource
def load_rebuilt_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model.h5")
    
    model = build_cnn_architecture()
    
    try:
        model.load_weights(model_path)
        return model, False
    except Exception as e:
        return model, True

model, is_simulation = load_rebuilt_model()

# ==========================================
# 3. SIDEBAR / SYSTEM CONTROL PANEL
# ==========================================
with st.sidebar:
    st.title("DeepSpill Core")
    st.caption("v2.1.0 • Convolutional Neural Network")
    st.divider()
    
    st.subheader("System Status")
    if is_simulation:
        st.warning("⚠️ Running in Demo Mode\n(Weights unverified/corrupted)")
    else:
        st.success("🟢 CNN Engine Online & Verified")
        
    st.info("💡 **Tip:** For best inference precision, upload high-resolution SAR (Synthetic Aperture Radar) imagery.")

# ==========================================
# 4. MAIN DASHBOARD INTERFACE
# ==========================================
st.title("🛢️ Marine Oil Spill Analytics Dashboard")
st.markdown("##### Real-time automated satellite & aerial imagery analysis for marine environmental protection.")
st.divider()

# Top row metrics
m1, m2, m3 = st.columns(3)
m1.metric("Engine Target Class", "Oil Spill vs Clear Water")
m2.metric("Input Dimension", "128 × 128 × 3 (RGB)")
m3.metric("Classification Threshold", "0.50 (Sigmoid)")

st.write("##")

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📥 Data Ingestion")
    uploaded_image = st.file_uploader(
        label="Drop satellite imagery file here", 
        type=["jpg", "jpeg", "png"],
        help="Supports PNG, JPG, JPEG formats."
    )
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        # FIX: Swapped to use_column_width=True for compatibility
        st.image(image, caption="Ingested Source Image", use_column_width=True)

with col2:
    st.subheader("📊 Analytics & Inference Summary")
    
    if uploaded_image is not None:
        with st.spinner("Executing forward pass through CNN layers..."):
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            resized_image = image.resize((128, 128))
            img_array = np.array(resized_image) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            prediction_prob = float(model.predict(img_batch)[0][0])
            
        threshold = 0.5
        is_spill = prediction_prob >= threshold
        confidence = prediction_prob * 100 if is_spill else (1 - prediction_prob) * 100
        
        if is_spill:
            st.error(f"### 🚨 THREAT DETECTED: Anomaly Class Identified as Oil Spill")
            st.write("**Threat Severity Level Indicator:**")
            st.progress(int(confidence))
        else:
            st.success(f"### ✅ ZONE CLEAR: No Marine Anomalies Detected")
            st.write("**Confidence Level:**")
            st.progress(int(confidence))
            
        st.write("---")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(
                label="Target Probability", 
                value=f"{confidence:.2f}%",
                delta="CRITICAL ANOMALY" if is_spill else "SAFE WATER",
                delta_color="inverse" if is_spill else "normal"
            )
        with res_col2:
            st.metric(label="Raw Network Output Value", value=f"{prediction_prob:.5f}")
            
        with st.expander("🛠️ Advanced Developer & Model Metadata"):
            st.json({
                "model_framework": "TensorFlow/Keras",
                "input_tensor_shape": list(img_batch.shape),
                "layer_count": len(model.layers),
                "activation_function": "Sigmoid (Output Layer)",
                "classification_cutoff": threshold,
                "inferred_verdict": "Spill Detected" if is_spill else "Clear"
            })
    else:
        st.info("Please ingest an image in the left panel to trigger the diagnostic sequence.")