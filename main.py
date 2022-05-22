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
row1_col1, row1_col2 = st.columns([1, 2]) #[1, 2] bezeichnet den Platz, welcher dem Graphen (links) und dem Text (rechts) zukommen soll. So 1/3 Graph, 2/3 Text 

#Header links
row1_col1.subheader("Ihr Resultat")

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
fig1, ax = plt.subplots(figsize = (8, 4))
ax.bar(1, berechneHeartDisease(), color = barcolor(berechneHeartDisease))
ax.set_ylabel("Risiko", fontsize = 18) #Ich hab das [%] rausgenommen, da das sonst missverstanden werden kann wenn yticks unter 1 sind -T
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.xticks([])
ax.tick_params(axis='y', which='major', labelsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
row1_col1.pyplot(fig1, use_container_width = True)

#Header rechts
row1_col2.subheader("Wie ist Ihr Risiko zu interpretieren?")

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
row1_col2.write(message1(berechneHeartDisease), use_container_width = True)

#############################################################################################################
#Funktion 2: Welche Faktoren haben den größten Einfluss? ####################################################

#Erstellen von 2 Spalten, die linke für den Graph, die rechte für eine automatisch generierte Interpretation
row2_col1, row2_col2 = st.columns([1, 1])

#Header links
row2_col1.subheader("Wie setzt sich das Risiko zusammen?")

#Header rechts
row2_col2.subheader("Welchen Anteil haben veränderbare Faktoren?")

#Die Funktionen mit welchen das Risiko von einzelnen Faktoren berechnet werden können
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
                  'Anteil_am_Risiko': [berechneRisikoVonBMI(), 
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
df_sorted = df_Excel_contents.sort_values('Anteil_am_Risiko', ascending = False)

###Seaborn-Plot

df_sorted = df_Excel_contents.sort_values('Anteil_am_Risiko', ascending = False)

fig2 = plt.figure(figsize=(8,4))
sns.barplot(data=df_sorted, 
            x = 'Anteil_am_Risiko', 
            y = 'Feature',
            palette = 'Spectral'
           )

row2_col1.pyplot(fig2, use_container_width = True)


#Plot veränderbarer Features:
#Vorbereitung der Daten
Excel_contents2 = { 'Feature' : ['BMI',
                                 'Rauchen',
                                 'Sport', 
                                 'Schlaf'
                                ],
                   'Anteil_am_Risiko': [berechneRisikoVonBMI(),
                                        berechneRisikoVonSmoking(),
                                        berechneRisikoVonSport(),
                                        berechneRisikoVonSchlaf(),
                                       ]
                  }

#Erstellung eines DataFrames
df_Excel_contents2 = pd.DataFrame(data=Excel_contents2)
df_sorted2 = df_Excel_contents2.sort_values('Anteil_am_Risiko', ascending = False)

#Seaborn-Plot 2

fig3 = plt.figure(figsize=(8,4))
sns.barplot(data = df_sorted2, 
            x = 'Anteil_am_Risiko',
            y = 'Feature',
            palette = 'Spectral'
           )

row2_col2.pyplot(fig3, use_container_width = True)

##############################################################################################
#Funktion 3: Wo ist der Website-Nutzer im Vergleich zu den Daten?#############################
#Erstellen von 2 Spalten
row3_col1, row3_col2 = st.columns([1, 1])

#Header links
row3_col1.subheader("Wo befindet sich Ihr Risiko im Vergleich?")

#Header rechts
#row3_col2.subheader(Ranking_Function())

#def Ranking_Function():
#  user_risk_Heart_Disease = berechneHeartDisease()
#  Risk_of_all_people_in_xTest = Log_Reg.predict_proba(xTest).copy()
#  Risk_of_all_people_in_xTest = Risk_of_all_people_in_xTest.values.tolist()
#  soon_to_be_sorted_List = []
#  for i in Risk_of_all_people_in_xTest:
#    soon_to_be_sorted_List += i
#  sorted_List = sorted(soon_to_be_sorted_List)
#  fancy_df = pd.DataFrame(sorted_List, columns = ['Probability_1'])
#  ranking = np.searchsorted(fancy_df['Probability_1'], user_risk_Heart_Disease, side = 'left')
#  ranking = (ranking/len(fancy_df))*100
#  return ranking # The Result is the percentage in full numbers (3 means 3%). Meaning, at 3%, 97% of people have a higher risk for heart disease

#Um einen Bug zu umgehen, mache ich hier eine neue Implementierung des train_test_split
#Vorbereitung train_test_split
af = pd.read_csv("heart_2020_cleaned.csv")
af = af.dropna()
af = af.copy().sample(40000, random_state=76856645)
af['HeartDisease']      = af.HeartDisease.replace({'Yes': 1, 'No': 0})
af['Smoking']           = af.Smoking.replace({'Yes': 1, 'No': 0})
af['AlcoholDrinking']   = af.AlcoholDrinking.replace({'Yes': 1, 'No': 0})
af['Stroke']            = af.Stroke.replace({'Yes': 1, 'No': 0})
af['DiffWalking']       = af.DiffWalking.replace({'Yes': 1, 'No': 0})
af['Sex']               = af.Sex.replace({'Male': 1, 'Female': 0})
af['Asthma']            = af.Asthma.replace({'Yes': 1, 'No': 0})
af['PhysicalActivity']  = af.PhysicalActivity.replace({'Yes': 1, 'No': 0})
af['KidneyDisease']     = af.KidneyDisease.replace({'Yes': 1, 'No': 0})
af['SkinCancer']        = af.SkinCancer.replace({'Yes': 1, 'No': 0})
af['Diabetic']          = af.Diabetic.replace({'Yes': 3, 'No': 0, 'No, borderline diabetes': 2, 'Yes (during pregnancy)':1})
af['GenHealth']         = af.GenHealth.replace({'Poor': 0, 'Fair': 1, 'Excellent': 2, 'Good': 3, 'G':3, 'Very good': 4})
af['AgeCategory']       = af.AgeCategory.replace({'18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4, '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9, '70-74': 10, '75-79':11, '80 or older':12})
af['Race']              = af.Race.replace({'American Indian/Alaskan Native': 0, 'Asian': 1, 'Black': 2, 'Hispanic': 3, 'Other': 4, 'White': 5})
#der train_test_split
y = af['HeartDisease']
x = af.copy().drop(columns='HeartDisease', axis=1) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

def Ranking_Function():
    user_risk_Heart_Disease = berechneHeartDisease()
    Risk_of_all_people_in_x_test = Log_Reg.predict_proba(x_test).copy() #Diese Linie ist wo die Probleme passieren!
    Risk_of_all_people_in_x_test = Risk_of_all_people_in_x_test.values.tolist()
    soon_to_be_sorted_List_of_x_test_proba = []
    for i in Risk_of_all_people_in_x_test:
      soon_to_be_sorted_List_of_x_test_proba += i
    #sorted_list_of_x_test_proba = sorted(soon_to_be_sorted_List_of_x_test_proba)
    #fancy_df = pd.DataFrame(sorted_list_of_x_test_proba, columns = ['Probability_1'])
    #percentile_of_user = np.searchsorted(fancy_df['Probability_1'], user_risk_Heart_Disease, side = 'left')
    return soon_to_be_sorted_List_of_x_test_proba#percentile_of_user #3 means 3%

row3_col2.subheader(Ranking_Function())
# weiß noch nicht, was man hier machen kann

##############################################################################################
#Funktion 4: Erstellen einer Verbindung zu Anbietern##########################################
#Erstellen von 3 Spalten
row4_col1, row4_col2, row4_col3 = st.columns([1, 1, 1]) #3 Möglichkeiten, auf Websiten zuzugreifen (z.B. Anti-Rauchen, Abnehmen und Schlaf- /Stressmanagement)

#Spalte links
row4_col1.subheader("Hilfe beim Abnehmen")
with row4_col1:
    button1 = st.button('Optionen Abnehmen')
if button1:
    row4_col1.markdown('Versicherung: [CSS-Versicherung](https://www.css.ch/de/privatkunden/meine-gesundheit/ernaehrung/gesund-abnehmen.html)  \n' + 
                       'Sport: [weightwatchers](https://www.weightwatchers.com/ch/de/blog/abnehmen?g_acctid=578-410-2929&g_adgroupid=131213429166&g_adid=576066783055&g_adtype=search&g_campaign=GE_WW_CH-DE_qdstw_qobjc_qbudc_qaudp_qrtgn_qpma_qostz_qdevz_qlobr_qgeon_qkwn&g_campaignid=11226104309&g_keyword=abnehmen&g_keywordid=kwd-44003010&g_network=g&gclid=Cj0KCQjwspKUBhCvARIsAB2IYuuVS2v2dR9a22u8y84x_mV9qO7KHGpA5qqYyorvUXO-R_-YZ0naegcaArI4EALw_wcB&gclsrc=aw.ds)  \n' + 
                       'Ernährung: [EatSmarter](https://eatsmarter.de/abnehmen/gesund-abnehmen/ernaehrungsplan-zum-abnehmen)')


#Spalte mitte
row4_col2.subheader("Mit Rauchen aufhören")
with row4_col2:
    button2 = st.button("Optionen Rauchen")
if button2:
    row4_col2.markdown('Infobroschüre: [Smokefree](https://www.smokefree.ch/de/wie-aufhoeren/selber-aufhoeren/#:~:text=Ersetze%20das%20Rauchen%20mit%20Bewegung,immer%20wieder%20f%C3%BCr%20deinen%20Rauchstopp.)  \n' +
                      'Schweizerische Herzstiftung: [swissheart](https://swissheart.ch/so-bleiben-sie-gesund/gesund-leben/rauchstopp)  \n' +
                      'Nikotinersatztherapie: [Nicorette](https://www.nicorette.de/raucherentwoehnung/rauchen-aufhoeren-tipps)')


#Spalte rechts
row4_col3.subheader("Jetzt besser schlafen")
with row4_col3:
    button3 = st.button("Optionen besser schlafen")
if button3:
    row4_col3.markdown('Übersicht: [atupri](https://www.atupri.ch/de/gesund-leben/wissen/psyche/schlafen)  \n' +
                      'Pharmazeutische Lösungen: [Valverde](https://www.valverde.ch/produkte/valverde-schlaf-und-schlaf-forte?gclid=Cj0KCQjwspKUBhCvARIsAB2IYuuVTie4PnhCWpMvMhvxjLqfA1MFtxZ0lEiYvKNWpE2-ShNG1hx2L0UaAl5bEALw_wcB)  \n' +
                      'Hintergründe und weitere Tipps: [Helios](https://www.helios-gesundheit.de/magazin/gesunder-schlaf/news/schlafhygiene-8-wertvolle-tipps-zum-einschlafen/)')
    
##########################################################################################
#Funktion 5: Downloaden einer Zusammenfassung der Ergebnisse##############################

#Aufbau
row5_col1, row5_col2 = st.columns([1, 1])

row5_col1.subheader("Hier können Sie Ihre Resultate downloaden")

#Aufnahme der Daten aus Funktion 2

#Erstellung eines DataFrames
df_Excel_contents = pd.DataFrame(data=Excel_contents)

csv_Excel_contents = df_Excel_contents.to_csv(index = False).encode('utf-8')

download_results = row5_col1.download_button(label = 'Ihre Resultate', 
                                             data = csv_Excel_contents, 
                                             file_name = 'Mein Resultat.csv', 
                                             mime = 'text/csv', 
                                             help = 'Hier links klicken zum Download als Excel-Datei', 
                                             key='download-csv')

#Extras
if download_results:
    row5_col1.markdown('Vielen Dank für das Nutzen der App, wir wünschen alles Gute!')

