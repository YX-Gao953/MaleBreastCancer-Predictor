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
    This model is designed to predict the 5-year overall survival rate for men with luminal early breast cancer and to stratify based on the prediction results. 
    
    Please enter the patient data in the input boxes below.
    
    Note: The first four variables assess the preconditions. If the requirements of the model are not met, you will receive a notification.
     
    This model is intended to ASSIST physicians in making clinical decisions ONLY.
""")

# Define input
ERstatus = st.selectbox("Estrogen receptor (ER) status:", options=['Positive', 'Negative', 'Unknown/Borderline'])
PRstatus = st.selectbox("Progesterone receptor (PR) status:", options=['Positive', 'Negative', 'Unknown/Borderline'])
HER2status = st.selectbox("Human epidermal growth factor receptor-2 (HER-2) status:", options=['Negative', 'Positive', 'Unknown/Borderline'])
Mstage = st.selectbox("M stage:", options=[0, 1])
Breastsurgery = st.selectbox("Breast surgery:", options=['No (Refused/Contraindicated)', 'Planned', 'Done (Mastectomy, with or without Reconstruction)', 'Done (Partial Mastectomy)'])

age = st.number_input("Age (supports 18-90y only):", min_value=18, max_value=90)
Tstage = st.selectbox("T stage:", options=[1, 2, 3, 4])
Nstage = st.selectbox("N stage:", options=[0, 1, 2, 3])
Histology = st.selectbox("Histology:", options=['Ductal', 'Other'])
Grade = st.selectbox("Grade:", options=['I', 'II', 'III/IV'])




##Form the input data
columns = ['ERstatus', 'PRstatus', 'HER2status', 'Mstage', 'Breast surgery',
           'Age', 
           'Histology=1', 'Histology=2', 
           'Grade=1', 'Grade=2', 'Grade=3', 
           'T stage=1', 'T stage=2', 'T stage=3', 'T stage=4', 
           'N stage=0', 'N stage=1', 'N stage=2', 'N stage=3']

input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)


### Fill the input data into empty dataframe
# Preconditions
if ERstatus == 'Positive':
    input_df['ERstatus'] = 1
elif ERstatus == 'Negative':
    input_df['ERstatus'] = 0
else: input_df['ERstatus'] = 999

if PRstatus == 'Positive':
    input_df['PRstatus'] = 1
elif PRstatus == 'Negative':
    input_df['PRstatus'] = 0
else: input_df['PRstatus'] = 999

if HER2status == 'Positive':
    input_df['HER2status'] = 1
elif HER2status == 'Negative':
    input_df['HER2status'] = 0
else: input_df['HER2status'] = 999

if Mstage == 1:
    input_df['Mstage'] = 1
else: input_df['Mstage'] = 0

if Breastsurgery == 'No (Refused/Contraindicated)':
    input_df['Breastsurgery'] = 0
elif Breastsurgery == 'Planned':
    input_df['Breastsurgery'] = 1
elif Breastsurgery == 'Done (Mastectomy, with or without Reconstruction)':
    input_df['Breastsurgery'] = 2
elif Breastsurgery == 'Done (Partial Mastectomy)':
    input_df['Breastsurgery'] = 3



# Predictors
input_df['Age'] = age
if Histology == 'Ductal':
    input_df['Histology=1'] = 1
elif Histology == 'Other':
    input_df['Histology=2'] = 1

if Grade == 'I':
    input_df['Grade=1'] = 1
elif Grade == 'II':
    input_df['Grade=2'] = 1
elif Grade == 'III/IV':
    input_df['Grade=3'] = 1

if Tstage == 1:
    input_df['T stage=1'] = 1
elif Tstage == 2:
    input_df['T stage=2'] = 1
elif Tstage == 3:
    input_df['T stage=3'] = 1
elif Tstage == 4:
    input_df['T stage=4'] = 1

if Nstage == 0:
    input_df['N stage=0'] = 1
elif Nstage == 1:
    input_df['N stage=1'] = 1
elif Nstage == 2:
    input_df['N stage=2'] = 1
elif Nstage == 3:
    input_df['N stage=3'] = 1



precondition_columns = ['ERstatus', 'PRstatus', 'HER2status', 'Mstage', 'Breastsurgery']
variables_columns = ['Age', 'Histology=1', 'Histology=2', 
                     'Grade=1', 'Grade=2', 'Grade=3', 
                     'T stage=1', 'T stage=2', 'T stage=3', 'T stage=4', 
                     'N stage=0', 'N stage=1', 'N stage=2', 'N stage=3']
input_df_precondition = input_df[precondition_columns]
input_df_variables = input_df[variables_columns]
input_df_variables = input_df_variables.applymap(lambda x: int(x) if isinstance(x, (int, float)) else x)

##Load the model and start prediction
model = joblib.load('CwGB_best_model.pkl')
cut_off_1 = 0.2285
advice_1 = ""
advice_2 = ""
advice_3 = ""
if st.button("Predict"):
    if (input_df_precondition['Mstage'].iloc[0] == 1 or
    (input_df_precondition['ERstatus'].iloc[0] != 1 and input_df_precondition['PRstatus'].iloc[0] != 1) or
    input_df_precondition['HER2status'].iloc[0] != 0):
        advice_1 = "Sorry, the model is not available for this patient."
        advice_2 = "It is indicated for men with ER and/or PR-positive, HER-2 negative early breast cancer."
        advice_3 = "Please select a proper candidate."
    elif input_df_precondition['Breastsurgery'].iloc[0] == 0 :
        advice_1 = "Sorry, the model is not available for this patient."
        advice_2 = "The breast radical surgery must be planned or done."
        advice_3 = "Please select a proper candidate."
    else:
        chf_funcs = model.predict_cumulative_hazard_function(input_df_variables.values, return_array=False)

        times = np.arange(12, 121, 6, dtype=int)

        risk_scores = np.row_stack([chf(times) for chf in chf_funcs])
        risk_scores_5y = risk_scores[:, 8][0]
        overall_survival_5y = np.exp(-risk_scores_5y) * 100

        if risk_scores_5y <= 0.2285:
            advice_1 = (f"The prediction result is LOW-RISK.")
            advice_2 = (f"It suggests NO significant overall survival benefit from chemotherapy.")
            advice_3 = (f"According to the training set of this model, the 5-year overall survival rate of LOW-RISK group is 87.0%.")
        else:
            advice_1 = (f"The prediction result is HIGH-RISK.")
            advice_2 = (f"In HIGH-RISK group, patients who received chemotherapy showed significantly higher overall survival rate compared to those who did not.")
            advice_3 = (f"According to the training set of this model, the 5-year overall survival rate of HIGH-RISK group is 65.5%.")
st.write(advice_1)
st.write(advice_2)
st.write(advice_3)