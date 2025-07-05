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
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis



##Define user input
# Streamlit user intersurface
st.title("Luminal Early Male Breast Cancer Risk Predictor")

#Add description
st.markdown("""
    This model is designed to predict the 5-year overall survival rate and to predict the benefit of chemotherapy.
            
    Candidates: MEN with luminal early breast cancer. A definitive surgery must be scheduled or performed.
    
    Please enter the patient data in the input boxes below.
    
    Note: If the preconditions of the model are not met, you will receive a notification.
     
    This model is intended to ASSIST physicians in making clinical decisions ONLY.
""")

# Define input
ERstatus = st.selectbox("Estrogen receptor (ER) status:", options=['Positive', 'Negative', 'Unknown/Borderline'])
PRstatus = st.selectbox("Progesterone receptor (PR) status:", options=['Positive', 'Negative', 'Unknown/Borderline'])
HER2status = st.selectbox("Human epidermal growth factor receptor-2 (HER-2) status:", options=['Negative', 'Positive', 'Unknown/Borderline'])
Mstage = st.selectbox("M stage:", options=[0, 1])
Breast_surgery = st.selectbox("Breast surgery:", options=['Not performed (e.g. contraindicated/refused)', 'Scheduled/Performed', 'Unknown'])

age = st.number_input("Age:", min_value=18, max_value=90)
Tstage = st.selectbox("T stage:", options=[1, 2, 3, 4])
Nstage = st.selectbox("N stage:", options=[0, 1, 2, 3])
Histology = st.selectbox("Histology:", options=['Ductal', 'Other'])
Grade = st.selectbox("Grade:", options=['I', 'II', 'III/IV'])
Radiotherapy = st.selectbox("Radiotherapy:", options=['No', 'Yes'])
Chemotherapy = 0


##Process the input data
columns = ['ERstatus', 'PRstatus', 'HER2status', 'Mstage', 'Breast_surgery'
           'Age', 
           'Histology=1', 'Histology=2', 
           'Grade=1', 'Grade=2', 'Grade=3', 
           'T stage=1', 'T stage=2', 'T stage=3', 'T stage=4', 
           'N stage=0', 'N stage=1', 'N stage=2', 'N stage=3', 
           'Radiotherapy=0', 'Radiotherapy=1',
           'Chemotherapy=0', 'Chemotherapy=1']

input_df_nochemo = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
input_df_nochemo['Chemotherapy=0'] = 1
input_df_nochemo['Chemotherapy=1'] = 0

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

if Breast_surgery == 'Not performed (e.g. contraindicated/refused)':
    input_df_nochemo['Breast_surgery'] = 0
elif Breast_surgery == 'Scheduled/Performed': 
    input_df_nochemo['Breast_surgery'] = 1
else: input_df_nochemo['Breast_surgery'] = 999


input_df_nochemo['Age'] = age
if Histology == 'Ductal':
    input_df_nochemo['Histology=1'] = 1
elif Histology == 'Other':
    input_df_nochemo['Histology=2'] = 1

if Grade == 'I':
    input_df_nochemo['Grade=1'] = 1
elif Grade == 'II':
    input_df_nochemo['Grade=2'] = 1
elif Grade == 'III/IV':
    input_df_nochemo['Grade=3'] = 1

if Tstage == 1:
    input_df_nochemo['T stage=1'] = 1
elif Tstage == 2:
    input_df_nochemo['T stage=2'] = 1
elif Tstage == 3:
    input_df_nochemo['T stage=3'] = 1
elif Tstage == 4:
    input_df_nochemo['T stage=4'] = 1

if Nstage == 0:
    input_df_nochemo['N stage=0'] = 1
elif Nstage == 1:
    input_df_nochemo['N stage=1'] = 1
elif Nstage == 2:
    input_df_nochemo['N stage=2'] = 1
elif Nstage == 3:
    input_df_nochemo['N stage=3'] = 1

if Radiotherapy == 0:
    input_df_nochemo['Radiotherapy=0'] = 1
elif Radiotherapy == 1:
    input_df_nochemo['Radiotherapy=1'] = 1


input_df_chemo = input_df_nochemo.copy()
input_df_chemo['Chemotherapy=0'] = 0
input_df_chemo['Chemotherapy=1'] = 1

precondition_columns = ['ERstatus', 'PRstatus', 'HER2status', 'Mstage', 'Breast_surgery']
variables_columns = ['Age', 'Histology=1', 'Histology=2', 
                     'Grade=1', 'Grade=2', 'Grade=3', 
                     'T stage=1', 'T stage=2', 'T stage=3', 'T stage=4', 
                     'N stage=0', 'N stage=1', 'N stage=2', 'N stage=3', 
                     'Radiotherapy=0', 'Radiotherapy=1',
                     'Chemotherapy=0', 'Chemotherapy=1']
input_df_precondition = input_df_nochemo[precondition_columns]
input_df_nochemo_variables = input_df_nochemo[variables_columns]
input_df_chemo_variables = input_df_chemo[variables_columns]

##Load the model and start prediction
model = joblib.load('CwGB_best_model.pkl')
cut_off = 0.18765

advice_1 = "Sorry, the model is not available for this patient."
advice_2 = "Only male patients with ER and/or PR-positive, HER-2 negative early breast cancer are indicated."
advice_3 = "It is designed for predicting the benefit of (NEO)ADJUVANT chemotherapy. SURGERY must be scheduled or performed."
advice_4 = "The prediction result is LOW-RISK. It suggests NO significant overall survival benefit from chemotherapy."
advice_5 = "The prediction result is HIGH-RISK. It suggests overall survival may be improved significantly by chemotherapy."
advice_6 = "The prediction result is HIGH-RISK. However, the overall survival may not be improved by chemotherapy."

if st.button("Predict"):
    if (input_df_precondition['Mstage'].iloc[0] == 1 or
    (input_df_precondition['ERstatus'].iloc[0] != 1 and input_df_precondition['PRstatus'].iloc[0] != 1) or
    input_df_precondition['HER2status'].iloc[0] != 0
    ):
        st.write(advice_1)
        st.write(advice_2)
    elif input_df_precondition['Breast_surgery'] != 1:
        st.write(advice_1)
        st.write(advice_3)
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

        chemo_benefit = overall_survival_5y_chemo - overall_survival_5y_nochemo
        if chemo_benefit < 0.1:
            chemo_benefit_str = "<0.1"
        else: 
            chemo_benefit_str = f"{chemo_benefit:.1f}"

        advice_7 = f"WITHOUT chemotherapy, the 5-year overall survival rate is predicted to be {overall_survival_5y_nochemo:.1f}%."
        advice_8 = f"Estimates of absolute 5-year overall survival rate benefit of chemotherapy is: {chemo_benefit_str}%."

        if risk_scores_5y_nochemo <= cut_off:
            st.write(advice_4)
            st.write(advice_7)
            st.write(advice_8)
        elif chemo_benefit <0.1:
            st.write(advice_6)
            st.write(advice_7)
            st.write(advice_8)
        else:
            st.write(advice_5)
            st.write(advice_7)
            st.write(advice_8)
