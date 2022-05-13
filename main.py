import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from Herz import Herz
import matplotlib.pyplot as plt
import numpy as np


#Global Variablen

Log_Reg = pickle.load(open('finalized_LogReg_model.sav', 'rb'))
manager = Herz()

###UI Aufbau

#Seite Config
st.set_page_config(
    page_title = 'Heart Health Assessment App',
    page_icon = '❤️',
    layout = 'wide'
    )
st.title('Heart Health Assessment App')

#Input Sliders/Selectboxes
st.sidebar.markdown('Bitte beantworten Sie die folgenden Fragen:')
körpergewicht = st.sidebar.slider("Was ist Ihr Körpergewicht", 30, 150, 70)
körpergrösse = st.sidebar.slider("Was ist Ihre Körpergrösse", 60, 250, 180)
schlaffZeit = st.sidebar.slider("Wie lange schlafen sie pro Abend?", 1, 14, 8)
raucher = st.sidebar.selectbox("Haben sie mehr als 100 Zigaretten in ihrem Leben geraucht?", options=["Ja", "Nein"])
alkohol = st.sidebar.selectbox("Trinken Sie mehr als 14 alkoholische Getränke pro Woche?", options=["Ja", "Nein"])
geschlecht = st.sidebar.selectbox("Geschlecht", options=["Männlich", "Weiblich"])
physicalHealth = st.sidebar.slider("Wie oft fühlten Sie sich in den vergangen 30 Tagen physisch nicht gut?", 0, 30, 5)
mentalHealth = st.sidebar.slider("Wie oft fühlten Sie sich in den vergangen 30 Tagen mental nicht gut?", 0, 30, 5)
schlaganfall = st.sidebar.selectbox("Hatten Sie jemals einen Schlaganfall", options=["Ja", "Nein"])
problemeBeimGehen = st.sidebar.selectbox("Haben Sie Probleme beim Gehen?", options=["Ja", "Nein"])
asthma = st.sidebar.selectbox("Haben Sie Asthma", options=["Ja", "Nein"])
physicalactivity = st.sidebar.selectbox("Haben Sie in den letzten 30 Tagen physisch trainiert", options=["Ja", "Nein"])
kidneyDisease = st.sidebar.selectbox("Hatten Sie jemals Nierenkrankheiten (exkl. Nierensteine/Niereninfektionen", options=["Ja", "Nein"])
skinCancer = st.sidebar.selectbox("Hatten Sie jemals Hautkrebs", options=["Ja", "Nein"])
diabetic = st.sidebar.selectbox("Haben Sie Diabetis", options=["Ja", "Nein"])
genHealth = st.sidebar.selectbox("Wie würden Sie ihre Gesundheit beschreiben", options=["Poor", "Fair", "Excellent", "Good", "Very Good"])
ageCategory = st.sidebar.selectbox("Wie alt sind Sie", options=["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"])
rasse = st.sidebar.selectbox("Was ist ihre ethnische Zugehörigkeit", options=["American Indian/Alaskan Native", "Asian", "Black", "Hispanic", "Other", "White"])


#Logistische Regression
def berechneHeartDisease():
    data = {
        "BMI": [körpergewicht/((körpergrösse/100)**2)],
        'Smoking': [manager.jaOderNein(raucher)],
        'AlcoholDrinking': [manager.jaOderNein(alkohol)],
        "Stroke": [manager.jaOderNein(schlaganfall)],
        "PhysicalHealth": [physicalHealth],
        "MentalHealth": [mentalHealth],
        "DiffWalking": [manager.jaOderNein(problemeBeimGehen)],
        "Sex": [manager.sexKonvertieren(geschlecht)],
        "AgeCategory": [manager.alterKonvertieren(ageCategory)],
        "Race": [manager.rasseKonvertieren(rasse)],
        "Diabetic": [manager.jaOderNein(diabetic)],
        "PhysicalActivity": [manager.jaOderNein(physicalactivity)],
        "GenHealth": [manager.genHealthKonvertieren(genHealth)],
        "SleepTime": [schlaffZeit],
        "Asthma": [manager.jaOderNein(asthma)],
        "KidneyDisease": [manager.jaOderNein(kidneyDisease)],
        "SkinCancer": [manager.jaOderNein(skinCancer)]
    }
    inputInfos = pd.DataFrame(data=data)
    heartdisease = Log_Reg.predict_proba(inputInfos)[0][1]
    return heartdisease

####Beginn Graph 1: Graphische Darstellung des Risikos###############################

#Erstellen von 2 Spalten, die linke für den Graph, die rechte für eine automatisch generierte Nachricht
row1_col1, row1_col2 = st.columns([1, 2])

row1_col1.subheader("Ihr Herzkrankheits-Risiko")
fig1, ax = plt.subplots(figsize = (8, 4))
ax.bar(heartdisease.index, heartdisease, color = "green")
row1_col1.pyplot(fig1, use_container_width = True)



row1_col2.subheader("Wie ist mein Risiko zu interpretieren?")



#st.write(berechneHeartDisease())
