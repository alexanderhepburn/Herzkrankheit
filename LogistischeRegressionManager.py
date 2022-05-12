import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from Herz import Herz

#Hier wird der Datensatz direkt aus collab eingelesen


x_train = pd.read_csv("x_train_heart_data.csv")
x_test  = pd.read_csv("x_test_heart_data.csv")
x_train = pd.read_csv("x_train_heart_data.csv")

path = '' #Path an EUER System anpassen!
df = pd.read_csv(path+'heart_2020_cleaned.csv')
df = pd.read_csv("heart_2020_cleaned.csv")
df = df.copy().sample(40000, random_state=42)

###############################

df['HeartDisease']      = df.HeartDisease.replace(          {'Yes': 1, 'No': 0})
df['Smoking']           = df.Smoking.replace(               {'Yes': 1, 'No': 0})
df['AlcoholDrinking']   = df.AlcoholDrinking.replace(       {'Yes': 1, 'No': 0})
df['Stroke']            = df.Stroke.replace(                {'Yes': 1, 'No': 0})
df['DiffWalking']       = df.DiffWalking.replace(           {'Yes': 1, 'No': 0})
df['Sex']               = df.Sex.replace(                   {'Male': 1, 'Female': 0})
df['Asthma']            = df.Asthma.replace(                {'Yes': 1, 'No': 0})
df['PhysicalActivity']  = df.PhysicalActivity.replace(      {'Yes': 1, 'No': 0})
df['KidneyDisease']     = df.KidneyDisease.replace(         {'Yes': 1, 'No': 0})
df['SkinCancer']        = df.SkinCancer.replace(            {'Yes': 1, 'No': 0})
df['Diabetic']          = df.Diabetic.replace(              {'Yes': 3, 'No': 0, 'No, borderline diabetes': 2, 'Yes (during pregnancy)':1})
df['GenHealth']         = df.GenHealth.replace(             {'Poor': 0, 'Fair': 1, 'Excellent': 2, 'Good': 3, 'Very good': 4})
df['AgeCategory']       = df.AgeCategory.replace(           {'18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4, '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9, '70-74': 10, '75-79':11, '80 or older':12})
df['Race']              = df.Race.replace(                  {'American Indian/Alaskan Native': 0, 'Asian': 1, 'Black': 2, 'Hispanic': 3, 'Other': 4, 'White': 5})


y = df['HeartDisease']
x = df.copy().drop(columns='HeartDisease', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
model = LogisticRegression(random_state=1, C=0.004291934260128779, class_weight='balanced', dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=3000, multi_class='auto', n_jobs=None, penalty='l2',  solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
Log_Reg = model.fit(x_train, y_train)