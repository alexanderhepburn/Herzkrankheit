import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from Herz import Herz
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.feature_selection import RFE
import webbrowser
import seaborn as sns
from streamlit.components.v1 import html

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

st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: .5rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 0.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
                .custom-h3 {
                    margin-top: -70px;
                    margin-bottom: -40px;
                }
                .custom-h6 {
                    margin-top: -30px;
                    margin-bottom: -30px;
                }
                .c-container {
                    text-align: center;
                    margin-bottom: 50px;
                }
                .c-t-h6 {
                    text-align: center; 
                }
                
                .c-t-h3 {
                    padding-top: 60px; 
                    text-align: center;               
                }
                
                .b-c {
                  font-size: 16px;
                  text-decoration: none;
                  background-color: #00bfbe;
                  text-color: white;
                  padding: 15px 20px 15px 20px;
                  border-radius: 12px;
                }
                
                .css-1adrfps {
                    padding-top: 40px !important;
                }
                
                .b-c:link {
                  color: white; 
                }
                
                .b-c:active {
                  color: white; 
                }
            
                .b-c:visited {
                  color: white; 
                }
                
                .b-c:hover {
                  color: white; 
                }
                
                .c-h3 {
                    margin-bottom: 14px;
                }
                
                @media screen and (max-width: 1600px) {
                    .custom-h3 {
                        margin-top: -50px;
                        font-size: 20px;
                    }
                    
                    .custom-h6 {
                        margin-top: -20px;
                        font-size: 14px;
                    }
                }
                @media screen and (max-width: 1280px) {
                    .custom-h3 {
                        margin-top: -50px;
                        font-size: 18px;
                    }
                    
                    .custom-h6 {
                        margin-top: -15px;
                        font-size: 13px;
                    }
                }  
                
                @media screen and (max-width: 1080px) {
                    .custom-h3 {
                        margin-top: -50px;
                        font-size: 16px;
                    }
                    
                    .custom-h6 {
                        margin-top: -5px;
                        font-size: 11px;
                    }
                }        
                
        </style>
        """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center'>Heart Health Assessment App</h1>", unsafe_allow_html=True)

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
genHealth = st.sidebar.selectbox("Wie würden Sie ihre Gesundheit beschreiben", options=["Poor", "Fair", "Good", "Very Good", "Excellent"])
ageCategory = st.sidebar.selectbox("Wie alt sind Sie", options=["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"])
rasse = st.sidebar.selectbox("Was ist ihre ethnische Zugehörigkeit", options=["White", "American Indian/Alaskan Native", "Asian", "Black", "Hispanic", "Other"])


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

#############################################################################################################
####Beginn Funktion 1: Graphische Darstellung des Risikos und passende Message###############################

#Erstellen von 2 Spalten, die linke für den Graph, die rechte für eine automatisch generierte Nachricht
row1_col1, row1_col2, row1_col3, row1_col4 = st.columns([1, 1, 1, 1]) #[1, 2] bezeichnet den Platz, welcher dem Graphen (links) und dem Text (rechts) zukommen soll. So 1/3 Graph, 2/3 Text



#Zuteilung der Farbe je nach Risiko
def barcolor(berechneHeartDisease):
    if berechneHeartDisease() < 0.25:
        colorcode = '#93c47d'
    elif 0.25 <= berechneHeartDisease() < 0.5:
        colorcode = '#ffd966'
    elif 0.5 <= berechneHeartDisease() < 0.75:
        colorcode = '#f6b26b'
    else:
        colorcode = '#cc4125'
    return colorcode


#Plot
# fig1, ax = plt.subplots(figsize = (8, 4))
# ax.bar(1, berechneHeartDisease(), color = barcolor(berechneHeartDisease))
# ax.set_ylabel("Risiko", fontsize = 18) #Ich hab das [%] rausgenommen, da das sonst missverstanden werden kann wenn yticks unter 1 sind -T
# plt.yticks([0, 0.25, 0.5, 0.75, 1])
# plt.xticks([])
# ax.tick_params(axis='y', which='major', labelsize=16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# row1_col1.pyplot(fig1, use_container_width = True)


#############################################################################################################
#Funktion 2: Welche Faktoren haben den größten Einfluss? ####################################################

#Erstellen von 2 Spalten, die linke für den Graph, die rechte für eine automatisch generierte Interpretation
#row2_col1, row2_col2 = st.columns([1, 1])

#Header links
#row2_col1.markdown("<h3 style='text-align: center'>Wie setzt sich das Risiko zusammen?</h3>", unsafe_allow_html=True)

#Header rechts
#row2_col2.markdown("<h3 style='text-align: center'>Welchen Anteil haben veränderbare Faktoren?</h3>", unsafe_allow_html=True)


#Die Funktionen mit welchen das Risiko von einzelnen Faktoren verglichen werden kann
user_Risiko = berechneHeartDisease()

def berechneRisikoVonBMI():
    data = {
        "BMI": 21,
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
    BMI_Risiko = user_Risiko-heartdisease
    return BMI_Risiko #{'BMI_Risiko':BMI_Risiko}


def berechneRisikoVonSmoking():
    data = {
        "BMI": [körpergewicht/((körpergrösse/100)**2)],
        'Smoking': 0, # 0 = 'Nein'
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
    Smoking_Risiko = user_Risiko-heartdisease
    return Smoking_Risiko #{'Smoking_Risiko':Smoking_Risiko}


def berechneRisikoVonAlkohol():
    data = {
        "BMI": [körpergewicht/((körpergrösse/100)**2)],
        'Smoking': [manager.jaOderNein(raucher)],
        'AlcoholDrinking': 1, # 1 = 'Ja'
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
    Smoking_Risiko = user_Risiko-heartdisease
    return Smoking_Risiko #{'Smoking_Risiko':Smoking_Risiko}


def berechneRisikoVonStroke():
    data = {
        "BMI": [körpergewicht/((körpergrösse/100)**2)],
        'Smoking': [manager.jaOderNein(raucher)],
        'AlcoholDrinking': [manager.jaOderNein(alkohol)],
        "Stroke": 1, # 1 = 'Ja'
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
    Stroke_Risiko = user_Risiko-heartdisease
    return Stroke_Risiko #{'Stroke_Risiko':Stroke_Risiko}


def berechneRisikoVonPhysicalHealth():
    data = {
        "BMI": [körpergewicht/((körpergrösse/100)**2)],
        'Smoking': [manager.jaOderNein(raucher)],
        'AlcoholDrinking': [manager.jaOderNein(alkohol)],
        "Stroke": [manager.jaOderNein(schlaganfall)],
        "PhysicalHealth": 14,
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
    PhysicalHealth_Risiko = user_Risiko-heartdisease
    return PhysicalHealth_Risiko #{'PhysicalHealth_Risiko':PhysicalHealth_Risiko}


def berechneRisikoVonMentalHealth():
    data = {
        "BMI": [körpergewicht/((körpergrösse/100)**2)],
        'Smoking': [manager.jaOderNein(raucher)],
        'AlcoholDrinking': [manager.jaOderNein(alkohol)],
        "Stroke": [manager.jaOderNein(schlaganfall)],
        "PhysicalHealth": [physicalHealth],
        "MentalHealth": 14,
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
    MentalHealth_Risiko = user_Risiko-heartdisease
    return MentalHealth_Risiko #{'MentalHealth_Risiko':MentalHealth_Risiko}


def berechneRisikoVonDiffWalking():
    data = {
        "BMI": [körpergewicht/((körpergrösse/100)**2)],
        'Smoking': [manager.jaOderNein(raucher)],
        'AlcoholDrinking': [manager.jaOderNein(alkohol)],
        "Stroke": [manager.jaOderNein(schlaganfall)],
        "PhysicalHealth": [physicalHealth],
        "MentalHealth": [mentalHealth],
        "DiffWalking": 0, # 0 = 'Nein'
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
    DiffWalking_Risiko = user_Risiko-heartdisease
    return DiffWalking_Risiko #{'DiffWalking_Risiko':DiffWalking_Risiko}


def berechneRisikoVonSex():
    data = {
        "BMI": [körpergewicht/((körpergrösse/100)**2)],
        'Smoking': [manager.jaOderNein(raucher)],
        'AlcoholDrinking': [manager.jaOderNein(alkohol)],
        "Stroke": [manager.jaOderNein(schlaganfall)],
        "PhysicalHealth": [physicalHealth],
        "MentalHealth": [mentalHealth],
        "DiffWalking": [manager.jaOderNein(problemeBeimGehen)],
        "Sex": 1,
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
    Sex_Risiko = user_Risiko-heartdisease
    return Sex_Risiko #{'Sex_Risiko':Sex_Risiko}


def berechneRisikoVonAgeCategory():
    data = {
        "BMI": [körpergewicht/((körpergrösse/100)**2)],
        'Smoking': [manager.jaOderNein(raucher)],
        'AlcoholDrinking': [manager.jaOderNein(alkohol)],
        "Stroke": [manager.jaOderNein(schlaganfall)],
        "PhysicalHealth": [physicalHealth],
        "MentalHealth": [mentalHealth],
        "DiffWalking": [manager.jaOderNein(problemeBeimGehen)],
        "Sex": [manager.sexKonvertieren(geschlecht)],
        "AgeCategory": 0,
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
    AgeCategory_Risiko = user_Risiko-heartdisease
    return AgeCategory_Risiko #{'AgeCategory_Risiko':AgeCategory_Risiko}


def berechneRisikoVonRace():
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
        "Race": 5,
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
    Race_Risiko = user_Risiko-heartdisease
    return Race_Risiko #{'Race_Risiko':Race_Risiko}


def berechneRisikoVonDiabetic():
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
        "Diabetic": 0, # 0 = 'Nein'
        "PhysicalActivity": [manager.jaOderNein(physicalactivity)],
        "GenHealth": [manager.genHealthKonvertieren(genHealth)],
        "SleepTime": [schlaffZeit],
        "Asthma": [manager.jaOderNein(asthma)],
        "KidneyDisease": [manager.jaOderNein(kidneyDisease)],
        "SkinCancer": [manager.jaOderNein(skinCancer)]
    }
    inputInfos = pd.DataFrame(data=data)
    heartdisease = Log_Reg.predict_proba(inputInfos)[0][1]
    Diabetic_Risiko = user_Risiko-heartdisease
    return Diabetic_Risiko #{'Diabetic_Risiko':Diabetic_Risiko}


def berechneRisikoVonSport():
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
        "PhysicalActivity": 1, #1 = 'Ja'
        "GenHealth": [manager.genHealthKonvertieren(genHealth)],
        "SleepTime": [schlaffZeit],
        "Asthma": [manager.jaOderNein(asthma)],
        "KidneyDisease": [manager.jaOderNein(kidneyDisease)],
        "SkinCancer": [manager.jaOderNein(skinCancer)]
    }
    inputInfos = pd.DataFrame(data=data)
    heartdisease = Log_Reg.predict_proba(inputInfos)[0][1]
    Sport_Risiko = user_Risiko-heartdisease
    return Sport_Risiko #{'Sport_Risiko':Sport_Risiko}


def berechneRisikoVonGenHealth():
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
        "GenHealth": 2,
        "SleepTime": [schlaffZeit],
        "Asthma": [manager.jaOderNein(asthma)],
        "KidneyDisease": [manager.jaOderNein(kidneyDisease)],
        "SkinCancer": [manager.jaOderNein(skinCancer)]
    }
    inputInfos = pd.DataFrame(data=data)
    heartdisease = Log_Reg.predict_proba(inputInfos)[0][1]
    GenHealth_Risiko = user_Risiko-heartdisease
    return GenHealth_Risiko #{'GenHealth_Risiko':GenHealth_Risiko}


def berechneRisikoVonSchlaf():
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
        "SleepTime": 9,
        "Asthma": [manager.jaOderNein(asthma)],
        "KidneyDisease": [manager.jaOderNein(kidneyDisease)],
        "SkinCancer": [manager.jaOderNein(skinCancer)]
    }
    inputInfos = pd.DataFrame(data=data)
    heartdisease = Log_Reg.predict_proba(inputInfos)[0][1]
    Schlaf_Risiko = user_Risiko-heartdisease
    return Schlaf_Risiko #{'Schlaf_Risiko':Schlaf_Risiko}


def berechneRisikoVonAsthma():
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
        "Asthma": 0, # 0 = 'Nein'
        "KidneyDisease": [manager.jaOderNein(kidneyDisease)],
        "SkinCancer": [manager.jaOderNein(skinCancer)]
    }
    inputInfos = pd.DataFrame(data=data)
    heartdisease = Log_Reg.predict_proba(inputInfos)[0][1]
    Asthma_Risiko = user_Risiko-heartdisease
    return Asthma_Risiko #{'Asthma_Risiko':Asthma_Risiko}


def berechneRisikoVonKidneyDisease():
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
        "KidneyDisease": 0, # 0 = 'Nein'
        "SkinCancer": [manager.jaOderNein(skinCancer)]
    }
    inputInfos = pd.DataFrame(data=data)
    heartdisease = Log_Reg.predict_proba(inputInfos)[0][1]
    KidneyDisease_Risiko = user_Risiko-heartdisease
    return KidneyDisease_Risiko 


def berechneRisikoVonSkinCancer():
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
        "SkinCancer": 0 # 0 = 'Nein'
    }
    inputInfos = pd.DataFrame(data=data)
    heartdisease = Log_Reg.predict_proba(inputInfos)[0][1]
    SkinCancer_Risiko = user_Risiko-heartdisease
    return SkinCancer_Risiko


fig, ax = plt.subplots()

size = 0.25
radius = 1
startangle = 230
t = .78
fontSize = 20.0

## Herzkrankheit Risiko

risiko = round(berechneHeartDisease() * 100, 1)

ax.pie([(risiko*t), 100-(risiko*t)], radius=radius, colors=[manager.farbeFuerPro(risiko), 'w'],
       wedgeprops=dict(width=size, edgecolor='w'), startangle=startangle, counterclock=False)
ax.text(0, 0, f'{risiko}%', ha='center', va='center', fontsize=fontSize)
ax.set(aspect="equal")
row1_col1.pyplot(fig, use_container_width = True)

row1_col1.markdown("<h3 style='text-align: center' class='custom-h3'>Herzkrankheit Risiko</h3>", unsafe_allow_html=True)
row1_col1.markdown("<h6 style='text-align: center' class='custom-h6'>Der angegebene Wert gibt die Wahrscheinlichkeit einer Herzkrankheit wieder</h6>", unsafe_allow_html=True)

## Risko im Vergleich

risiko = round(manager.rankingFunction(berechneHeartDisease(), logReg=Log_Reg), 1)

fig2, ax2 = plt.subplots()
ax2.pie([(risiko*t), 100-(risiko*t)], radius=radius, colors=[manager.farbeFuerPro(risiko), 'w'],
       wedgeprops=dict(width=size, edgecolor='w'), startangle=startangle, counterclock=False)
ax2.text(0, 0, f'{risiko}%', ha='center', va='center', fontsize=fontSize)
ax2.set(aspect="equal")
row1_col2.pyplot(fig2, use_container_width = True)
row1_col2.markdown("<h3 style='text-align: center' class='custom-h3'>Risiko im Vergleich</h3>", unsafe_allow_html=True)
row1_col2.markdown(f"<h6 style='text-align: center' class='custom-h6'>Das für Sie berechnete Risiko ist höher als das von {risiko} % Personen aus den Testdaten</h6>", unsafe_allow_html=True)


## BMI Anteil

cat = manager.berechneCategorie(berechneRisikoVonBMI())

fig3, ax3 = plt.subplots()

ax3.pie([(cat/5)*100*t, 100-((cat/5)*100*t)], radius=radius, colors=[manager.farbeFuerCat(cat), 'w'],
       wedgeprops=dict(width=size, edgecolor='w'), startangle=startangle, counterclock=False)
ax3.text(0, 0, f'{cat}', ha='center', va='center', fontsize=fontSize)
ax3.set(aspect="equal")
row1_col3.pyplot(fig3, use_container_width = True)
row1_col3.markdown("<h3 style='text-align: center' class='custom-h3'>BMI</h3>", unsafe_allow_html=True)
if cat == 1:
    row1_col3.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr BMI wurde in Kategorie 1 eingestuft. Dies bedeutet, dass der Anteil des BMI am Risiko bei unter 5 % liegt.</h6>", unsafe_allow_html=True)
if cat == 2:
    row1_col3.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr BMI wurde in Kategorie 2 eingestuft. Dies bedeutet, dass der Anteil des BMI am Risiko zwischen 5 % und 10 % liegt.</h6>", unsafe_allow_html=True)
if cat == 3:
    row1_col3.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr BMI wurde in Kategorie 3 eingestuft. Dies bedeutet, dass der Anteil des BMI am Risiko zwischen 10 % und 15 % liegt.</h6>", unsafe_allow_html=True)
if cat == 4:
    row1_col3.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr BMI wurde in Kategorie 4 eingestuft. Dies bedeutet, dass der Anteil des BMI am Risiko zwischen 15 % und 20 % liegt.</h6>", unsafe_allow_html=True)
if cat == 5:
    row1_col3.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr BMI wurde in Kategorie 5 eingestuft. Dies bedeutet, dass der Anteil des BMI am Risiko über 20 % liegt.</h6>", unsafe_allow_html=True)


## Sport Anteil

cat = manager.berechneCategorie(berechneRisikoVonSport())

fig4, ax4 = plt.subplots()

ax4.pie([(cat/5)*100*t, 100-((cat/5)*100*t)], radius=radius, colors=[manager.farbeFuerCat(cat), 'w'],
       wedgeprops=dict(width=size, edgecolor='w'), startangle=startangle, counterclock=False)
ax4.text(0, 0, f'{cat}', ha='center', va='center', fontsize=fontSize)
ax4.set(aspect="equal")
row1_col4.pyplot(fig4, use_container_width = True)
row1_col4.markdown("<h3 style='text-align: center' class='custom-h3'>Sport</h3>", unsafe_allow_html=True)
if cat == 1:
    row1_col4.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr Anteil sportlicher Betätigung wurde in Kategorie 1 eingestuft. Dies bedeutet, dass der Anteil mangelhafter sportlicher Betätigung am Risiko bei unter 5 % liegt.</h6>", unsafe_allow_html=True)
if cat == 2:
    row1_col4.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr Anteil sportlicher Betätigung wurde in Kategorie 2 eingestuft. Dies bedeutet, dass der Anteil mangelhafter sportlicher Betätigung am Risiko zwischen 5 % und 10 % liegt.</h6>", unsafe_allow_html=True)
if cat == 3:
    row1_col4.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr Anteil sportlicher Betätigung wurde in Kategorie 3 eingestuft. Dies bedeutet, dass der Anteil mangelhafter sportlicher Betätigung am Risiko zwischen 10 % und 15 % liegt.</h6>", unsafe_allow_html=True)
if cat == 4:
    row1_col4.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr Anteil sportlicher Betätigung wurde in Kategorie 4 eingestuft. Dies bedeutet, dass der Anteil mangelhafter sportlicher Betätigung am Risiko zwischen 15 % und 20 % liegt.</h6>", unsafe_allow_html=True)
if cat == 5:
    row1_col4.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr Anteil sportlicher Betätigung wurde in Kategorie 5 eingestuft. Dies bedeutet, dass der Anteil mangelhafter sportlicher Betätigung am Risiko über 20 % liegt.</h6>", unsafe_allow_html=True)


row2_col1, row2_col2, row2_col3, row2_col4 = st.columns([1, 1, 1, 1])



## Schlaf Anteil

cat = manager.berechneCategorie(berechneRisikoVonSchlaf())

fig5, ax5 = plt.subplots()

ax5.pie([(cat/5)*100*t, 100-((cat/5)*100*t)], radius=radius, colors=[manager.farbeFuerCat(cat), 'w'],
       wedgeprops=dict(width=size, edgecolor='w'), startangle=startangle, counterclock=False)
ax5.text(0, 0, f'{cat}', ha='center', va='center', fontsize=fontSize)
ax5.set(aspect="equal")
row2_col1.pyplot(fig5, use_container_width = True)
row2_col1.markdown("<h3 style='text-align: center' class='custom-h3'>Schlaf</h3>", unsafe_allow_html=True)
if cat == 1:
    row2_col1.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr Anteil von Schlaf wurde in Kategorie 1 eingestuft. Dies bedeutet, dass der Anteil mangelhaften Schlafes am Risiko bei unter 5 % liegt.</h6>", unsafe_allow_html=True)
if cat == 2:
    row2_col1.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr Anteil von Schlaf wurde in Kategorie 2 eingestuft. Dies bedeutet, dass der Anteil mangelhaften Schlafes am Risiko zwischen 5 % und 10 % liegt.</h6>", unsafe_allow_html=True)
if cat == 3:
    row2_col1.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr Anteil von Schlaf wurde in Kategorie 3 eingestuft. Dies bedeutet, dass der Anteil mangelhaften Schlafes am Risiko zwischen 10 % und 15 % liegt.</h6>", unsafe_allow_html=True)
if cat == 4:
    row2_col1.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr Anteil von Schlaf wurde in Kategorie 4 eingestuft. Dies bedeutet, dass der Anteil mangelhaften Schlafes am Risiko zwischen 15 % und 20 % liegt.</h6>", unsafe_allow_html=True)
if cat == 5:
    row2_col1.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr Anteil von Schlaf wurde in Kategorie 5 eingestuft. Dies bedeutet, dass der Anteil mangelhaften Schlafes am Risiko über 20 % liegt.</h6>", unsafe_allow_html=True)

## Mentale Gesundheit Anteil

cat = manager.berechneCategorie(berechneRisikoVonMentalHealth() * 16)

fig6, ax6 = plt.subplots()

ax6.pie([(cat/5)*100*t, 100-((cat/5)*100*t)], radius=radius, colors=[manager.farbeFuerCat(cat), 'w'],
       wedgeprops=dict(width=size, edgecolor='w'), startangle=startangle, counterclock=False)
ax6.text(0, 0, f'{cat}', ha='center', va='center', fontsize=fontSize)
ax6.set(aspect="equal")
row2_col2.pyplot(fig6, use_container_width = True)
row2_col2.markdown("<h3 style='text-align: center' class='custom-h3'>Mentale Gesundheit</h3>", unsafe_allow_html=True)
if cat == 1:
    row2_col2.markdown("<h6 style='text-align: center' class='custom-h6'>Ihre mentale Gesundheit wurde in Kategorie 1 eingestuft. Dies bedeutet, dass der Anteil mentaler Gesundheit am Risiko bei unter 5 % liegt.</h6>", unsafe_allow_html=True)
if cat == 2:
    row2_col2.markdown("<h6 style='text-align: center' class='custom-h6'>Ihre mentale Gesundheit wurde in Kategorie 2 eingestuft. Dies bedeutet, dass der Anteil mentaler Gesundheit am Risiko zwischen 5 % und 10 % liegt.</h6>", unsafe_allow_html=True)
if cat == 3:
    row2_col2.markdown("<h6 style='text-align: center' class='custom-h6'>Ihre mentale Gesundheit wurde in Kategorie 3 eingestuft. Dies bedeutet, dass der Anteil mentaler Gesundheit am Risiko zwischen 10 % und 15 % liegt.</h6>", unsafe_allow_html=True)
if cat == 4:
    row2_col2.markdown("<h6 style='text-align: center' class='custom-h6'>Ihre mentale Gesundheit wurde in Kategorie 4 eingestuft. Dies bedeutet, dass der Anteil mentaler Gesundheit am Risiko zwischen 15 % und 20 % liegt.</h6>", unsafe_allow_html=True)
if cat == 5:
    row2_col2.markdown("<h6 style='text-align: center' class='custom-h6'>Ihre mentale Gesundheit wurde in Kategorie 5 eingestuft. Dies bedeutet, dass der Anteil mentaler Gesundheit am Risiko über 20 % liegt.</h6>", unsafe_allow_html=True)

## Physiche Gesundheit Anteil

cat = manager.berechneCategorie(berechneRisikoVonPhysicalHealth() * 13)

fig7, ax7 = plt.subplots()

ax7.pie([(cat/5)*100*t, 100-((cat/5)*100*t)], radius=radius, colors=[manager.farbeFuerCat(cat), 'w'],
       wedgeprops=dict(width=size, edgecolor='w'), startangle=startangle, counterclock=False)
ax7.text(0, 0, f'{cat}', ha='center', va='center', fontsize=fontSize)
ax7.set(aspect="equal")
row2_col3.pyplot(fig7, use_container_width = True)
row2_col3.markdown("<h3 style='text-align: center' class='custom-h3'>Physiche Gesundheit</h3>", unsafe_allow_html=True)
if cat == 1:
    row2_col3.markdown("<h6 style='text-align: center' class='custom-h6'>Ihre physische Gesundheit wurde in Kategorie 1 eingestuft. Dies bedeutet, dass der Anteil physischer Gesundheit am Risiko bei unter 5 % liegt.</h6>", unsafe_allow_html=True)
if cat == 2:
    row2_col3.markdown("<h6 style='text-align: center' class='custom-h6'>Ihre physische Gesundheit wurde in Kategorie 2 eingestuft. Dies bedeutet, dass der Anteil physischer Gesundheit am Risiko zwischen 5 % und 10 % liegt.</h6>", unsafe_allow_html=True)
if cat == 3:
    row2_col3.markdown("<h6 style='text-align: center' class='custom-h6'>Ihre physische Gesundheit wurde in Kategorie 3 eingestuft. Dies bedeutet, dass der Anteil physischer Gesundheit am Risiko zwischen 10 % und 15 % liegt.</h6>", unsafe_allow_html=True)
if cat == 4:
    row2_col3.markdown("<h6 style='text-align: center' class='custom-h6'>Ihre physische Gesundheit wurde in Kategorie 4 eingestuft. Dies bedeutet, dass der Anteil physischer Gesundheit am Risiko zwischen 15 % und 20 % liegt.</h6>", unsafe_allow_html=True)
if cat == 5:
    row2_col3.markdown("<h6 style='text-align: center' class='custom-h6'>Ihre physische Gesundheit wurde in Kategorie 5 eingestuft. Dies bedeutet, dass der Anteil physischer Gesundheit am Risiko über 20 % liegt.</h6>", unsafe_allow_html=True)

## Alkohol Anteil

cat = 0
if alkohol == "Ja":
    cat = 3
else:
    cat = 1

fig8, ax8 = plt.subplots()

ax8.pie([(cat/5)*100*t, 100-((cat/5)*100*t)], radius=radius, colors=[manager.farbeFuerCat(cat), 'w'],
       wedgeprops=dict(width=size, edgecolor='w'), startangle=startangle, counterclock=False)
ax8.text(0, 0, f'{cat}', ha='center', va='center', fontsize=fontSize)
ax8.set(aspect="equal")
row2_col4.pyplot(fig8, use_container_width = True)
row2_col4.markdown("<h3 style='text-align: center' class='custom-h3'>Alkohol</h3>", unsafe_allow_html=True)
if cat == 1:
    row2_col4.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr Alkoholkonsum wurde in Kategorie 1 eingestuft. Dies bedeutet, dass der Anteil des Alkoholkonsums am Risiko bei unter 5 % liegt.</h6>", unsafe_allow_html=True)
if cat == 2:
    row2_col4.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr Alkoholkonsum wurde in Kategorie 2 eingestuft. Dies bedeutet, dass der Anteil des Alkoholkonsums am Risiko zwischen 5 % und 10 % liegt.</h6>", unsafe_allow_html=True)
if cat == 3:
    row2_col4.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr Alkoholkonsum wurde in Kategorie 3 eingestuft. Dies bedeutet, dass der Anteil des Alkoholkonsums am Risiko zwischen 10 % und 15 % liegt.</h6>", unsafe_allow_html=True)
if cat == 4:
    row2_col4.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr Alkoholkonsum wurde in Kategorie 4 eingestuft. Dies bedeutet, dass der Anteil des Alkoholkonsums am Risiko zwischen 15 % und 20 % liegt.</h6>", unsafe_allow_html=True)
if cat == 5:
    row2_col4.markdown("<h6 style='text-align: center' class='custom-h6'>Ihr Alkoholkonsum wurde in Kategorie 5 eingestuft. Dies bedeutet, dass der Anteil des Alkoholkonsums am Risiko über 20 % liegt.</h6>", unsafe_allow_html=True)

row999_col1 = st.columns(1)
row999_col1[0].markdown("<h3 class='c-t-h3'>Zusammenfassung</h3>", unsafe_allow_html=True)
if risiko > 50:
    row999_col1[0].markdown("<h6 class='c-t-h6'>Insgesamt scheint Ihr Risiko, an einer Herzkrankheit zu leiden, im Vergleich zu Durchschnitt erhöht. Wir raten Ihnen daher, die weiteren Funktionen dieser App zu nutzen, um Ihr Risiko effektiv zu senken, und bei Unwohlsein ärztlichen Rat aufzusuchen. Generelle Informationen zu den Kategorien: Kategorie 1 bedeutet, dass der Anteil am Risiko unter 5 % liegt, Kategorie 2 dass der Anteil zwischen 5 % und 10 % liegt, Kategorie 3 dass der Anteil zwischen 10 % und 15 % liegt, Kategorie 4 dass der Anteil zwischen 15 % und 20 % liegt und Kategorie 5 dass der Anteil über 20% liegt.</h6>", unsafe_allow_html=True)
if risiko <=50:
    row999_col1[0].markdown("<h6 class='c-t-h6'>Ihr Risiko, an einer Herzkrankheit zu leiden, ist kleiner oder gleich dem Durchschnitt. Gerne können Sie präventiv die weiteren Funktionen dieser App nutzen, um Ihr Risiko weiter zu senken. Bitte beachten Sie jedoch, dass das berechntete Ergebnis nicht bedeutet, dass Sie an keiner Herzkrankheit leiden! Wir raten Ihnen daher, bei Unwohlsein unbedingt ärztlichen Rat zu suchen. Generelle Informationen zu den Kategorien: Kategorie 1 bedeutet, dass der Anteil am Risiko unter 5 % liegt, Kategorie 2 dass der Anteil zwischen 5 % und 10 % liegt, Kategorie 3 dass der Anteil zwischen 10 % und 15 % liegt, Kategorie 4 dass der Anteil zwischen 15 % und 20 % liegt und Kategorie 5 dass der Anteil über 20% liegt.</h6>", unsafe_allow_html=True)

    
    

#Header rechts
#row1_col3.markdown("<h3 style='text-align: center'>Wie ist Ihr Risiko zu interpretieren?</h3>", unsafe_allow_html=True)
#Format des Risikos in % und gerundet:
HeartRiskInPerc = berechneHeartDisease()*100
HeartRisk = round(HeartRiskInPerc)

#Definierung der Message je nach Ergebnis:
def message1(berechneHeartDisease):
    if berechneHeartDisease() < 0.25:
        message = 'Das sieht sehr gut aus! Ihr Risiko, an einer Herzerkrankung zu leiden, liegt bei unter 25 %'
    elif 0.25 <= berechneHeartDisease() < 0.5:
        message = 'Kein Grund zur Sorge, Ihr Risiko liegt bei {}%. Wenn Sie Ihr Risiko weiter reduzieren wollen, empfehlen wir einen Blick auf unsere weiteren Funktionen weiter unten.'.format(HeartRisk)
    elif 0.5 <= berechneHeartDisease() < 0.75:
        message = 'Ihr Risiko liegt bei {}%. Bitte nutzen Sie unsere unten beigefügten Funktionen, um Ihr Risiko effektiv zu senken.'.format(HeartRisk)
    else:
        message = 'Ihr Risiko, an einer Herzerkrankung zu leiden, liegt bei über 75 %. Bitte informieren Sie sich weiter unten, wie Sie das Risiko senken können, und suchen Sie bei Unwohlsein ärztlichen Rat.'
    return message

#Aufruf der Nachricht
#row1_col3.write(message1(berechneHeartDisease), use_container_width = True)



stat1, stat2, stat3, stat4 = st.columns([1, 1, 1, 1])









#Plot aller Features:
Excel_contents = { 'Feature' : ['BMI', 
                             'Rauchen', 
                             'Alkoholkonsum', 
                             'Schlaganfall', 
                             'Koerperliche Gesundheit', 
                             'Mentale Gesundheit', 'Gehschwierigkeiten', 
                             'Geschlecht', 
                             'Alter', 
                             'Ethnie', 
                             'Diabetes', 
                             'Sport', 
                             'Generelles Wohlbefinden', 
                             'Schlaf', 'Asthma', 'Nierenkrankheiten', 'Hautkrebs'],
                  'Anteil am Risiko': [berechneRisikoVonBMI(), 
                                       berechneRisikoVonSmoking(), 
                                       berechneRisikoVonAlkohol(), 
                                       berechneRisikoVonStroke(), 
                                       berechneRisikoVonPhysicalHealth(), 
                                       berechneRisikoVonMentalHealth(), 
                                       berechneRisikoVonDiffWalking(), 
                                       berechneRisikoVonSex(), 
                                       berechneRisikoVonAgeCategory(), 
                                       berechneRisikoVonRace(), 
                                       berechneRisikoVonDiabetic(), 
                                       berechneRisikoVonSport(), 
                                       berechneRisikoVonGenHealth(), 
                                       berechneRisikoVonSchlaf(), 
                                       berechneRisikoVonAsthma(), 
                                       berechneRisikoVonKidneyDisease(), 
                                       berechneRisikoVonSkinCancer()]
                 }

#Erstellung eines DataFrames
df_Excel_contents = pd.DataFrame(data=Excel_contents)
df_sorted = df_Excel_contents.sort_values('Anteil am Risiko', ascending = False)

###Seaborn-Plot

df_sorted = df_Excel_contents.sort_values('Anteil am Risiko', ascending = False)

fig2 = plt.figure(figsize=(8,4))
sns.barplot(data=df_sorted, 
            x = 'Anteil am Risiko', 
            y = 'Feature',
            palette = 'Spectral'
           ).set(ylabel = ' ')

col10, col20 = st.columns([1, 1])

col10.pyplot(fig2, use_container_width = True)


#Plot veränderbarer Features:
#Vorbereitung der Daten
Excel_contents2 = { 'Feature' : ['BMI',
                                 'Mentale Gesundheit',
                                 'Sport', 
                                 'Schlaf'
                                ],
                   'Anteil am Risiko': [berechneRisikoVonBMI(),
                                        berechneRisikoVonMentalHealth(),
                                        berechneRisikoVonSport(),
                                        berechneRisikoVonSchlaf(),
                                       ]
                  }

#Erstellung eines DataFrames
df_Excel_contents2 = pd.DataFrame(data=Excel_contents2)
df_sorted2 = df_Excel_contents2.sort_values('Anteil am Risiko', ascending = False)

#Seaborn-Plot 2

fig3 = plt.figure(figsize=(8,4))
sns.barplot(data = df_sorted2, 
            x = 'Anteil am Risiko',
            y = 'Feature',
            palette = 'Spectral'
           ).set(ylabel = ' ')

col20.pyplot(fig3, use_container_width = True)

##############################################################################################
#Funktion 4: Erstellen einer Verbindung zu Anbietern##########################################
#Erstellen von 3 Spalten
row4_col1, row4_col2, row4_col3, row4_col4 = st.columns([1, 1, 1, 1]) #3 Möglichkeiten, auf Websiten zuzugreifen (z.B. Anti-Rauchen, Abnehmen und Schlaf- /Stressmanagement)


#Spalte links
row4_col1.markdown("<h3 class='c-h3' style='text-align: center'>Hilfe beim Abnehmen</h3>", unsafe_allow_html=True)
row4_col1.markdown("<div class='c-container'><a href='https://eatsmarter.de/abnehmen/gesund-abnehmen/ernaehrungsplan-zum-abnehmen' style='text-align: center' class='b-c'>Optionen Abnehmen</button></a>", unsafe_allow_html=True)

#Spalte mitte
row4_col2.markdown("<h3 class='c-h3' style='text-align: center'>Mentale Gesundheit</h3>", unsafe_allow_html=True)
row4_col2.markdown("<div class='c-container'><a href='https://www.bag.admin.ch/bag/de/home/strategie-und-politik/politische-auftraege-und-aktionsplaene/politische-auftraege-im-bereich-psychische-gesundheit.html' style='text-align: center' class='b-c'>Optionen mentale Gesundheit</a></div>", unsafe_allow_html=True)

#Spalte rechts
row4_col3.markdown("<h3 class='c-h3' style='text-align: center'>Jetzt besser schlafen</h3>", unsafe_allow_html=True)
row4_col3.markdown("<div class='c-container'><a href='https://www.helios-gesundheit.de/magazin/gesunder-schlaf/news/schlafhygiene-8-wertvolle-tipps-zum-einschlafen/' style='text-align: center' class='b-c'>Optionen besser schlafen</a></div>", unsafe_allow_html=True)


#Spalte rechts
row4_col4.markdown("<h3 class='c-h3' style='text-align: center'>Sport Informationen</h3>", unsafe_allow_html=True)
row4_col4.markdown("""<div class='c-container'><a href='https://www.ausdauerblog.de/mit-sport-anfangen/' style='text-align: center' class='b-c'>Mit Sport anfangen</a></div>""", unsafe_allow_html=True)

##########################################################################################
#Funktion 5: Downloaden einer Zusammenfassung der Ergebnisse##############################

#Extras

if st.sidebar.checkbox("Extras zeigen"):

    row5_col1, row5_col2 = st.columns([1, 1])

    row5_col1.subheader("Ihre Resultate herunterladen")

    # Aufnahme der Daten aus Funktion 2

    # Erstellung eines DataFrames
    
    #Hinzufügung von percentile und Gesamtrisiko:
    Excel_contents = Excel_contents.copy()
    Excel_contents['Feature'].insert(0, 'Risiko Percentile')
    Excel_contents['Anteil am Risiko'].insert(0, round(manager.rankingFunction(berechneHeartDisease(), logReg=Log_Reg), 1))
    Excel_contents['Feature'].insert(0, 'Gesamtrisiko')
    Excel_contents['Anteil am Risiko'].insert(0, round(berechneHeartDisease() * 100, 1))
    #Ende meiner Modifikationen
    
    df_Excel_contents = pd.DataFrame(data=Excel_contents)

    csv_Excel_contents = df_Excel_contents.to_csv(index=False).encode('utf-8')

    download_results = row5_col1.download_button(label='Ihre Resultate',
                                                 data=csv_Excel_contents,
                                                 file_name='Mein Resultat.csv',
                                                 mime='text/csv',
                                                 help='Hier links klicken zum Download als Excel-Datei',
                                                 key='download-csv')

    if download_results:
        row5_col1.markdown('Vielen Dank für das Nutzen der App, wir wünschen alles Gute!')

    row5_col2.subheader("Ihre Tabelle hochladen")
    uploaded_file = row5_col2.file_uploader("Excel-Tabelle hochladen:")
    if uploaded_file is not None:
         dataframe = pd.read_csv(uploaded_file)
         prediction = pd.DataFrame(Log_Reg.predict_proba(dataframe)).loc[:, 1]
         dataframe["HeartDiseaseRisk"] = prediction
         row5_col2.download_button('HeartDiseaseRisk Datei herunterladen', dataframe.to_csv(), "heartdiseaserisk.csv", "text/csv")
