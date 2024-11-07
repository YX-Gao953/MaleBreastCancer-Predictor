import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn import set_config
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

import pickle

from sksurv.preprocessing import OneHotEncoder, encode_categorical
from sksurv.ensemble import GradientBoostingSurvivalAnalysis



##Define user input
# Streamlit user intersurface
st.title("Hormone Receptor-Positive Early Male Breast Cancer Risk Predictor")

#Add description
st.markdown("""
    This model is designed to predict the 5-year overall survival rate for male patients with hormone receptor-positive early breast cancer and to stratify based on the prediction results. 
    
    Please enter the patient data in the input boxes below.
    
    Note: Although HER-2 status is not considered when constructing the model, it is still recommended for use only in predicting HER-2 negative patients. The first four variables assess the preconditions; if the requirements of the model are not met, you will receive a corresponding notification.
    
    If the prediction result indicates LOW-RISK, this suggests that chemotherapy may not improve the patient's prognosis. Therefore, chemotherapy omission may be considered.
            
    This model is intended to ASSIST physicians in making judgments.
""")

# Define input
ERstatus = st.selectbox("Estrogen receptor (ER) status:", options=['Positive', 'Negative', 'Unknown/Borderline'])
PRstatus = st.selectbox("Progesterone receptor (PR) status:", options=['Positive', 'Negative', 'Unknown/Borderline'])
HER2status = st.selectbox("Human epidermal growth factor receptor-2 (HER-2) status:", options=['Negative', 'Positive', 'Unknown/Borderline'])
Mstage = st.selectbox("M stage:", options=[0, 1])

age = st.number_input("Age:", min_value=18, max_value=100)
Tstage = st.selectbox("T stage:", options=[1, 2, 3, 4])
Nstage = st.selectbox("N stage:", options=[0, 1, 2, 3])
Histology = st.selectbox("Histology:", options=['Ductal', 'Other'])
Grade = st.selectbox("Grade:", options=['I', 'II', 'III/IV'])
Breastsurgery = st.selectbox("Breast surgery:", options=['No', 'Mastectomy', 'Partial mastectomy'])



##Form the input data
columns = ['ERstatus', 'PRstatus', 'HER2status', 'Mstage', 'Age', 'Histology=2', 'Grade=2', 'Grade=3', 'T stage=2', 'T stage=3',
           'T stage=4', 'N stage=1', 'N stage=2', 'N stage=3', 'Breast surgery=1',
           'Breast surgery=2', 'Chemotherapy=1']

input_df_nochemo = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

# Fill the input data into empty dataframe
if ERstatus == 'Positive':
    input_df_nochemo['ERstatus'] = 1
elif ERstatus == 'Negative':
    input_df_nochemo['ERstatus'] = 0
else: input_df_nochemo['ERstatus'] = 999

if PRstatus == 'Positive':
    input_df_nochemo['PRstatus'] = 1
elif PRstatus == 'Negative':
    input_df_nochemo['PRstatus'] = 0
else: input_df_nochemo['PRstatus'] = 999

if HER2status == 'Positive':
    input_df_nochemo['HER2status'] = 1
elif HER2status == 'Negative':
    input_df_nochemo['HER2status'] = 0
else: input_df_nochemo['HER2status'] = 999

if Mstage == 1:
    input_df_nochemo['Mstage'] = 1
else: input_df_nochemo['Mstage'] = 0

input_df_nochemo['Age'] = age
if Histology == 'Other':
    input_df_nochemo['Histology=2'] = 1
if Grade == 'II':
    input_df_nochemo['Grade=2'] = 1
elif Grade == 'III/IV':
    input_df_nochemo['Grade=3'] = 1
if Tstage == 2:
    input_df_nochemo['T stage=2'] = 1
elif Tstage == 3:
    input_df_nochemo['T stage=3'] = 1
elif Tstage == 4:
    input_df_nochemo['T stage=4'] = 1
if Nstage == 1:
    input_df_nochemo['N stage=1'] = 1
elif Nstage == 2:
    input_df_nochemo['N stage=2'] = 1
elif Nstage == 3:
    input_df_nochemo['N stage=3'] = 1
if Breastsurgery == 'Mastectomy':
    input_df_nochemo['Breast surgery=1'] = 1
elif Breastsurgery == 'Partial mastectomy':
    input_df_nochemo['Breast surgery=2'] = 1

input_df_chemo = input_df_nochemo
input_df_chemo['Chemotherapy=1'] = 1

precondition_columns = ['ERstatus', 'PRstatus', 'HER2status', 'Mstage']
variables_columns = ['Age', 'Histology=2', 'Grade=2', 'Grade=3', 
                     'T stage=2', 'T stage=3', 'T stage=4', 
                     'N stage=1', 'N stage=2', 'N stage=3', 
                     'Breast surgery=1', 'Breast surgery=2', 
                     'Chemotherapy=1']
input_df_precondition = input_df_nochemo[precondition_columns]
input_df_nochemo_variables = input_df_nochemo[variables_columns]
input_df_chemo_variables = input_df_chemo[variables_columns]

##Load the model and start prediction
model = joblib.load('Gradient_boosting_best_model.pkl')
cut_off = 0.21830
advice = ""
if st.button("Predict"):
    if (input_df_precondition['Mstage'].iloc[0] == 1 or
    (input_df_precondition['ERstatus'].iloc[0] != 1 and input_df_precondition['PRstatus'].iloc[0] != 1) or
    input_df_precondition['HER2status'].iloc[0] != 0):
        advice = '''The model is not available for this patient.
        It is indicated for men with ER and/or PR-positive, HER-2 negative early breast cancer.'''
    else:
        chf_funcs_nochemo = model.predict_cumulative_hazard_function(input_df_nochemo_variables.values, return_array=False)
        chf_funcs_chemo = model.predict_cumulative_hazard_function(input_df_chemo_variables.values, return_array=False)

        times = np.arange(12, 121, 6, dtype=int)

        risk_scores_nochemo = np.row_stack([chf(times) for chf in chf_funcs_nochemo])
        risk_scores_5y_nochemo = risk_scores_nochemo[:, 8][0]
        overall_survival_5y_nochemo = np.exp(-risk_scores_5y_nochemo) * 100

        risk_scores_chemo = np.row_stack([chf(times) for chf in chf_funcs_chemo])
        risk_scores_5y_chemo = risk_scores_chemo[:, 8][0]
        overall_survival_5y_chemo = np.exp(-risk_scores_5y_chemo) * 100
        if risk_scores_5y_nochemo <= cut_off:
            advice = f'The prediction result is LOW-RISK. It suggests the patient may NOT achieve significant overall survival benefit from chemotherapy.\nWITHOUT chemotherapy, the 5-year overall survival rate is predicted to be {overall_survival_5y_nochemo:.1f}%.\nWith chemotherapy, the 5-year overall survival rate is predicted to be {overall_survival_5y_chemo:.1f}%.'
        else:
            advice = f'The prediction result is HIGH-RISK. It suggests the patient may achieve significant overall survival benefit from chemotherapy.\nWITHOUT chemotherapy, the 5-year overall survival rate is predicted to be {overall_survival_5y_nochemo:.1f}%.\nWith chemotherapy, the 5-year overall survival rate is predicted to be {overall_survival_5y_chemo:.1f}%.'
st.write(advice)
