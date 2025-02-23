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
    st.subheader("UPLOADED IMAGE")
    ecg_user_image_read = ecg.getImage(uploaded_file)
    st.image(ecg_user_image_read)

    st.subheader("GRAY SCALE IMAGE")
    ecg_user_gray_image_read = ecg.GrayImgae(ecg_user_image_read)
    with st.expander("Gray SCALE IMAGE"):
        st.image(ecg_user_gray_image_read)

    st.subheader("DIVIDING LEADS")
    dividing_leads = ecg.DividingLeads(ecg_user_image_read)
    with st.expander("DIVIDING LEAD"):
        st.image('Leads_1-12_figure.png')
        st.image('Long_Lead_13_figure.png')

    st.subheader("PREPROCESSED LEADS")
    ecg_preprocessed_leads = ecg.PreprocessingLeads(dividing_leads)
    with st.expander("PREPROCESSED LEAD"):
        st.image('Preprossed_Leads_1-12_figure.png')
        st.image('Preprossed_Leads_13_figure.png')

    st.subheader("EXTRACTING SIGNALS(1-12)")
    ec_signal_extraction = ecg.SignalExtraction_Scaling(dividing_leads)
    with st.expander("CONTOUR LEADS"):
        st.image('Contour_Leads_1-12_figure.png')

    st.subheader("CONVERTING TO 1D SIGNAL")
    ecg_1dsignal = ecg.CombineConvert1Dsignal()
    with st.expander("1D Signals"):
        st.write(ecg_1dsignal)

    # Check if 1D signal is correctly shaped
    if isinstance(ecg_1dsignal, list):
        ecg_1dsignal = np.array(ecg_1dsignal)
    
    if len(ecg_1dsignal.shape) == 1:
        ecg_1dsignal = ecg_1dsignal.reshape(1, -1)  # Ensure 2D input for PCA
    
    st.write(f"Shape of 1D Signal: {ecg_1dsignal.shape}")

    # Load PCA model and apply dimensionality reduction
    st.subheader("PERFORM DIMENSIONALITY REDUCTION")
    pca_model_path = './Deployment/PCA_ECG.pkl'
    
    if not os.path.exists(pca_model_path):
        st.error(f"Model file '{pca_model_path}' not found. Please upload it.")
        st.stop()
    
    try:
        pca_loaded_model = joblib.load(pca_model_path)
        st.write("PCA Model Loaded Successfully.")

        if ecg_1dsignal.shape[1] < pca_loaded_model.n_components_:
            st.error(f"Input signal has {ecg_1dsignal.shape[1]} features, but PCA expects at least {pca_loaded_model.n_components_}.")
            st.stop()

        ecg_final = pca_loaded_model.transform(ecg_1dsignal)
        with st.expander("Dimensional Reduction"):
            st.write(ecg_final)
    except Exception as e:
        st.error(f"Error in dimensionality reduction: {e}")
        st.stop()
    
    # Pass to pre-trained ML model for prediction
    st.subheader("PASS TO PRETRAINED ML MODEL FOR PREDICTION")
    try:
        ecg_model = ecg.ModelLoad_predict(ecg_final)
        with st.expander("PREDICTION"):
            st.write(ecg_model)
    except Exception as e:
        st.error(f"Error in model prediction: {e}")
