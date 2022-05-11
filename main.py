#DER MOMENTANE STAND DER APP
#ANLEITUNG: LADE DIE DREI NEUEN DATEIEN HERUNTER, TU SIE IN DEN GLEICHEN ORDNER WIE 
#DIE APP DATEI, FOLGE DEM VIDEO TUTORIAL FALLS STREAMLIT NOCH NICHT INSTALLIERT 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas import DataFrame
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import pickle
################
################

#Loading of the data
#@st.cache()
path    = '' #Dieser Abschnitt muss an euer lokales System angepasst werden
x_train = pd.read_csv(path + "x_train_heart_data.csv")
x_test  = pd.read_csv(path + "x_test_heart_data.csv")
x_train = pd.read_csv(path + "x_train_heart_data.csv")
Log_Reg = pickle.load(open(path + 'finalized_LogReg_model.sav', 'rb'))
################
################

#General Look of the App
st.set_page_config(
    page_title = 'Heart Health Assessment App',
    page_icon = '❤️',
    layout = 'wide'
    )
st.title('Heart Health Assessment App')


körpergewicht = st.sidebar.slider("Geben Sie Ihr Körpergewicht ein", 30, 150, 70)
körpergrösse = st.sidebar.slider("Geben Sie Ihre Körpergrösse ein", 60, 250, 180)
raucher = st.sidebar.selectbox("Rauchen Sie?", options=["Ja", "Nein"])
alkohol = st.sidebar.selectbox("Trinken Sie regelmässig?", options=["Ja", "Nein"])
geschlecht = st.sidebar.selectbox("Geschlecht", options=["Männlich", "Weiblich"])
schlaganfall = st.sidebar.selectbox("Schlaganfall", options=["Ja", "Nein"])
asthma = st.sidebar.selectbox("Asthma", options=["Ja", "Nein"])
physicalactivity = st.sidebar.selectbox("Physical Activity", options=["Ja", "Nein"])


st.markdown('Bitte beantworten Sie die angegebene Fragen auf der linken Seite!')



#Hier wird der Datensatz direkt aus collab eingelesen
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
model = LogisticRegression()
Log_Reg = model.fit(x_train, y_train)

def jaOderNein(input):
    if input == "Ja":
        return 1
    else:
        return 0

def berechneHeartDisease():
    data = {
        "BMI": [körpergewicht/((körpergrösse/100)**2)],
        'Smoking': [jaOderNein(raucher)],
        'AlcoholDrinking': [jaOderNein(alkohol)],
        "Stroke": [jaOderNein(schlaganfall)],
        "PhysicalHealth": [0.0],
        "MentalHealth": [0.0],
        "Diffwalking": [1],
        "Sex": [0],
        "AgeCategory": [8],
        "Race": [5],
        "Diabetic": [1],
        "PhysicalActivity": [0],
        "GenHealth": [4],
        "SleepTime": [8.000],
        "Asthma": [0],
        "KidneyDisease": [0],
        "SkinCancer": [1]
    }
    inputInfos = pd.DataFrame(data=data)
    st.write(inputInfos)
    heartdisease = model.predict_proba(inputInfos)[0][1]
    return heartdisease


st.write(berechneHeartDisease())