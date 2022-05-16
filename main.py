import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from Herz import Herz
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

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
ax.set_ylabel("Risiko [%]")
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.xticks([])
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


#Funktion 2: Welche Faktoren haben den größten Einfluss? ####################################################

#Erstellen von 2 Spalten, die linke für den Graph, die rechte für eine automatisch generierte Interpretation
row2_col1, row2_col2 = st.columns([1, 1])

#Header links
row2_col1.subheader("Welche 5 Faktoren ausgenommen des Alters sind am wichtigsten?")

#Header rechts
row2_col2.subheader("Welche konkreten Maßnahmen können helfen?")
# hier dachte ich daran, Nachrichten zu generieren, wie etwa: Gesunde Ernährung, Sport, weniger Rauchen etc.

#Variablen-Bedeutung -> Hier erstmal vom Kurs rauskopiert (4 - Trees, Forests, Ensembles)

def plot_variable_importance(model, inputInfos):
    imp=DataFrame({"imp":model.feature_importances_, "names":inputInfos.columns}).sort_values("imp", ascending=True)
    fig2, ax = plt.subplots(figsize=(imp.shape[0]/6,imp.shape[0]/5), dpi=300)
    ax.barh(imp["names"],imp["imp"], color="#93c47d") 
    ax.set_xlabel('\nBedeutung der Variablen')
    ax.set_ylabel('Features\n') 
    ax.set_title('Bedeutung der Variablen - Abbildung\n') 
    plt.show()

#row2_col1.pyplot(plot_variable_importance(model, inputInfos), use_container_width = True)
