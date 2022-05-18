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
ax.set_ylabel("Risiko") #Ich hab das [%] rausgenommen, da das sonst missverstanden werden kann wenn yticks unter 1 sind -T
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

#############################################################################################################
#Funktion 2: Welche Faktoren haben den größten Einfluss? ####################################################

#Erstellen von 2 Spalten, die linke für den Graph, die rechte für eine automatisch generierte Interpretation
row2_col1, row2_col2 = st.columns([1, 1])

#Header links
row2_col1.subheader("Welche 5 Faktoren ausgenommen des Alters sind am wichtigsten?")

#Header rechts
row2_col2.subheader("Welche konkreten Maßnahmen können helfen?")
# hier dachte ich daran, Nachrichten zu generieren, wie etwa: Gesunde Ernährung, Sport, weniger Rauchen etc.

#Ich versuche mal etwas unelegantes hier
#NOTE: Wir haben nur 4 Effekte, welche effektiv einfach beeinflusst werden können, deswegen potentiell nur die top 1-2 Effekte darstellen
#bzw, wenn das GesamtRisiko unter 25% ist, dann vielleicht sagen, dass das Risiko nicht weiter verringert werden muss?
#Der Output von dieser Monstrosität ist ganz unten in biggest_risk_factor(). 
user_Risiko = berechneHeartDisease()

def berechneRisikoVonBMI():
    data = {
        "BMI": 20,
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
    return {'BMI_Risiko':BMI_Risiko}

def berechneRisikoVonAlkohol():
    data = {
        "BMI": [körpergewicht/((körpergrösse/100)**2)],
        'Smoking': [manager.jaOderNein(raucher)],
        'AlcoholDrinking': 'Nein',
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
    return {'Smoking_Risiko':Smoking_Risiko}


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
        "PhysicalActivity": 'Ja',
        "GenHealth": [manager.genHealthKonvertieren(genHealth)],
        "SleepTime": [schlaffZeit],
        "Asthma": [manager.jaOderNein(asthma)],
        "KidneyDisease": [manager.jaOderNein(kidneyDisease)],
        "SkinCancer": [manager.jaOderNein(skinCancer)]
    }
    inputInfos = pd.DataFrame(data=data)
    heartdisease = Log_Reg.predict_proba(inputInfos)[0][1]
    Sport_Risiko = user_Risiko-heartdisease
    return {'Sport_Risiko':Sport_Risiko}


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
    return {'Schlaf_Risiko':Schlaf_Risiko}

#Diese Funktion funktioniert noch nicht ganz. Eine Idee?
#def biggest_risk_factor():
#    biggest_risk_factorsss = berechneRisikoVonBMI()
#    for x in [berechneRisikoVonAlkohol(), berechneRisikoVonSport(), berechneRisikoVonSchlaf()]:
#        if x.values() > biggest_risk_factorsss.values():
#            biggest_risk_factorsss = x
#    return biggest_risk_factorsss #Output ist ein dic mit dem Risiko Factor und dem Risiko im 0.XX Format

#row2_col1.write(berechneRisikoVonSchlaf()) #Test obd dies funktioniert hat


def all_risks():
    for i in berechneRisikoVonBMI(), berechneRisikoVonSport(), berechneRisikoVonAlkohol(), berechneRisikoVonSchlaf():
        i.append(all_risks())
    return all_risks()

#y_achse = np.arange(len(brfs))

#fig2, ax = plt.subplots(figsize = (8, 4))
#ax.barh(y_pos, brfs, align='center')
#ax.set_yticks(y_pos, labels=brfs)
#ax.invert_yaxis()  # labels read top-to-bottom
#ax.set_xlabel('Anteil am Risiko')
#ax.set_title('Welche Variablen sind für Sie am wichtigsten?')

#row2_col1.pyplot(fig2, use_container_width = True) #Test obd dies funktioniert hat



##############################################################################################
#Funktion 3: Wo ist der Website-Nutzer im Vergleich zu den Daten?#############################
#Erstellen von 2 Spalten
row3_col1, row3_col2 = st.columns([1, 1])

#Header links
row3_col1.subheader("Wo befindet sich Ihr Risiko im Vergleich?")

#Header rechts
row3_col2.subheader("action title?")
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


