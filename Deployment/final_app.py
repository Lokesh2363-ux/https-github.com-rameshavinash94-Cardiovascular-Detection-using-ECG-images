import streamlit as st
import os
from Ecg import ECG

# Initialize ECG object
ecg = ECG()

# Get the uploaded image
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
    with st.expander(label='1D Signals'):
        st.write(ecg_1dsignal)
    
    """#### **PERFORM DIMENSIONALITY REDUCTION**"""
    pca_model_path = './Deployment/PCA_ECG.pkl'  # Ensure correct path
    
    if not os.path.exists(pca_model_path):
        st.error(f"Model file '{pca_model_path}' not found. Please upload it.")
    else:
        ecg_final = ecg.DimensionalReduciton(ecg_1dsignal)
        with st.expander(label='Dimensional Reduction'):
            st.write(ecg_final)
    
    """#### **PASS TO PRETRAINED ML MODEL FOR PREDICTION**"""
    ecg_model = ecg.ModelLoad_predict(ecg_final)
    with st.expander(label='PREDICTION'):
        st.write(ecg_model)
