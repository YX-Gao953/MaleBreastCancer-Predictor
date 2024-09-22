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
st.title("Male Breast Cancer Risk Predictor")

# Define input
age = st.number_input("Age:", min_value=18, max_value=100)
Tstage = st.selectbox("T stage:", options=[1, 2, 3, 4])
Nstage = st.selectbox("N stage:", options=[0, 1, 2, 3])
Histology = st.selectbox("Histology:", options=[1, 2])
Grade = st.selectbox("Grade:", options=[1, 2, 3])
Breastsurgery = st.selectbox("Breast surgery:", options=[0, 1, 2])
Chemotherapy = st.selectbox("Chemotherapy:", options=[0, 1])



##Form the input data
columns = ['Age', 'Histology=2', 'Grade=2', 'Grade=3', 'T stage=2', 'T stage=3',
           'T stage=4', 'N stage=1', 'N stage=2', 'N stage=3', 'Breast surgery=1',
           'Breast surgery=2', 'Chemotherapy=1']

input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

# Fill the input data into empty dataframe
input_df['Age'] = age
if Histology == 2:
    input_df['Histology=2'] = 1
if Grade == 2:
    input_df['Grade=2'] = 1
elif Grade == 3:
    input_df['Grade=3'] = 1
if Tstage == 2:
    input_df['T stage=2'] = 1
elif Tstage == 3:
    input_df['T stage=3'] = 1
elif Tstage == 4:
    input_df['T stage=4'] = 1
if Nstage == 1:
    input_df['N stage=1'] = 1
elif Nstage == 2:
    input_df['N stage=2'] = 1
elif Nstage == 3:
    input_df['N stage=3'] = 1
if Breastsurgery == 1:
    input_df['Breast surgery=1'] = 1
elif Breastsurgery == 2:
    input_df['Breast surgery=2'] = 1
if Chemotherapy == 1:
    input_df['Chemotherapy=1'] = 1



##Load the model and start prediction
model = joblib.load('Gradient boosting_best model.pkl')
cut_off = 0.21830

if st.button("Predict"):
    chf_funcs = model.predict_cumulative_hazard_function(input_df.values, return_array=False)
    times = np.arange(12, 121, 6, dtype=int)
    risk_scores = np.row_stack([chf(times) for chf in chf_funcs])
    risk_scores_5y = risk_scores[:, 8]
    if risk_scores_5y <= cut_off:
        advice = 'The risk stratification result is LOW-RISK'
    else:
        advice = 'The risk stratification result is HIGH-RISK'

    st.write(advice)