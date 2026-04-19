#  -*- coding: utf-8 -*-
# @Author  :   WEI
# @File    : run.py
# @Time    : 2024/3/25 21:46
# @Software: PyCharm
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import joblib
from pathlib import Path

st.markdown("""
<style>
    body {
        font-family: 'Arial', sans-serif;
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Neuroblastoma Distant Metastasis Detection Assistant")
st.write("Hi, this is a simple program about neuroblastoma discrimination.")
st.write("You can enter the relevant variables and get a preliminary discriminatory result.")
st.write("Our results are not the truth, and intelligently assist you in the discrimination of the likelihood of "
         "cancer metastasis.")
st.write("The truth needs to be analyzed and summarized by professionals from the perspective of mechanism and so on.")
st.write('  ')
st.write('  ')
st.write('  ')

age = st.sidebar.selectbox('Age', ['≥35', '＜35'])
Sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
Race = st.sidebar.selectbox('Race', ['White', 'Black', 'Other'])
Marital_status = st.sidebar.selectbox('Marital status', ['Married', 'Single (never married)', 'Other'])
site = st.sidebar.selectbox('Primary Site', ['Upper-outer quadrant', 'Upper-inner quadrant', 'Lower-outer quadrant', 'Lower-inner quadrant', 'Central portion', 'Other'])
T_stage = st.sidebar.selectbox('T stage', ['T1', 'T2', 'T3', 'T4'])
N_stage = st.sidebar.selectbox('N stage', ['N0', 'N1', 'N2', 'N3'])
Subtype = st.sidebar.selectbox('Subtype', ['HR+/HER2-', 'HR+/HER2+', 'HR-/HER2-', 'HR-/HER2+'])

data = [age, Sex, Race, Marital_status, site, T_stage, N_stage, Subtype]
cols = ['Age(years)', 'Sex', 'Race', 'Marital status', 'Primary site', 'T stage', 'N stage', 'Subtype']

X_test = pd.DataFrame([data], columns=cols)

project_root = Path(__file__).resolve().parent
model_path = project_root / "model" / "rf_model.joblib"
pipeline = joblib.load(model_path)

if st.sidebar.button("Predict"):
    y_prob = pipeline.predict_proba(X_test)[:, 1][0]
    y_pred = (y_prob >= 0.5).astype(int)

    if y_pred:
        st.markdown("""  
        <span style="color: black; font-size: 50px; background-color: #FFFF00;">  
        Probability of distant metastasis: {:.0f}% 
        </span>  
        """.format(y_prob * 100), unsafe_allow_html=True)
    else:
        st.markdown("""  
        <span style="color: black; font-size: 50px; background-color: #63B8FF;">  
        Probability of distant metastasis: {:.0f}% 
        </span>  
        """.format(y_prob * 100), unsafe_allow_html=True)
