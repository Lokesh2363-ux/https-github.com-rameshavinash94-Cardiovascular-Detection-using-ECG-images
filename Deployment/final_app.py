import streamlit as st
import os
import joblib
import numpy as np
from Ecg import ECG

# Initialize ECG object
ecg = ECG()

# File uploader
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    """#### **UPLOADED IMAGE**"""
    ecg_user_image_read = ecg.getImage(uploaded_file)
    st.image(ecg_user_image_read)

    """#### **GRAY SCALE IMAGE**"""
    ecg_user_gray_image_read = ecg.GrayImgae(ecg_user_image_read)
    with st.expander(label='Gray SCALE IMAGE'):
        st.image(ecg_user_gray_image_read)

    """#### **DIVIDING LEADS**"""
    dividing_leads = ecg.DividingLeads(ecg_user_image_read)
    with st.expander(label='DIVIDING LEAD'):
        st.image('Leads_1-12_figure.png')
        st.image('Long_Lead_13_figure.png')

    """#### **PREPROCESSED LEADS**"""
    ecg_preprocessed_leads = ecg.PreprocessingLeads(dividing_leads)
    with st.expander(label='PREPROCESSED LEAD'):
        st.image('Preprossed_Leads_1-12_figure.png')
        st.image('Preprossed_Leads_13_figure.png')

    """#### **EXTRACTING SIGNALS(1-12)**"""
    ec_signal_extraction = ecg.SignalExtraction_Scaling(dividing_leads)
    with st.expander(label='CONTOUR LEADS'):
        st.image('Contour_Leads_1-12_figure.png')

    """#### **CONVERTING TO 1D SIGNAL**"""
    ecg_1dsignal = ecg.CombineConvert1Dsignal()
    
    # Debugging Check: Ensure 1D signal is not empty
    st.write("1D Signal Type:", type(ecg_1dsignal))
    st.write("1D Signal Shape:", getattr(ecg_1dsignal, 'shape', 'No shape attribute'))

    if ecg_1dsignal is None or (hasattr(ecg_1dsignal, 'size') and ecg_1dsignal.size == 0):
        st.error("Error: 1D Signal is empty or invalid.")
        st.stop()

    with st.expander(label='1D Signals'):
        st.write(ecg_1dsignal)

    """#### **PERFORM DIMENSIONALITY REDUCTION**"""
    pca_model_path = './Deployment/PCA_ECG.pkl'  # Ensure correct path

    # Check if PCA model exists
    if not os.path.exists(pca_model_path):
        st.error(f"Model file '{pca_model_path}' not found. Please upload it.")
        st.stop()

    try:
        # Load PCA model
        pca_loaded_model = joblib.load(pca_model_path)
        st.write("PCA Model Loaded Successfully.")

        # Ensure input matches PCA model expectation
        expected_shape = pca_loaded_model.components_.shape[1]
        st.write("Expected Input Shape for PCA:", expected_shape)

        if isinstance(ecg_1dsignal, np.ndarray) and len(ecg_1dsignal.shape) == 1:
            ecg_1dsignal = ecg_1dsignal.reshape(1, -1)  # Reshape if 1D

        if ecg_1dsignal.shape[1] != expected_shape:
            st.error(f"Mismatch in input shape! Expected: {expected_shape}, Got: {ecg_1dsignal.shape[1]}")
            st.stop()

        # Apply PCA transformation
        ecg_final = ecg.DimensionalReduciton(ecg_1dsignal)
        st.write("Dimensionality Reduction Successful")

        with st.expander(label='Dimensional Reduction'):
            st.write(ecg_final)
    
    except Exception as e:
        st.error(f"Error in dimensionality reduction: {e}")
        st.stop()

    """#### **PASS TO PRETRAINED ML MODEL FOR PREDICTION**"""
    try:
        ecg_model = ecg.ModelLoad_predict(ecg_final)
        with st.expander(label='PREDICTION'):
            st.write(ecg_model)
    except Exception as e:
        st.error(f"Error in model prediction: {e}")
