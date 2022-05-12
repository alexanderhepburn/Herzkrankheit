import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from Herz import Herz

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
körpergewicht = st.sidebar.slider("Geben Sie Ihr Körpergewicht ein", 30, 150, 70)
körpergrösse = st.sidebar.slider("Geben Sie Ihre Körpergrösse ein", 60, 250, 180)
schlaffZeit = st.sidebar.slider("Wie lange schlaffen sie pro Abend?", 1, 14, 8)
raucher = st.sidebar.selectbox("Rauchen Sie?", options=["Ja", "Nein"])
alkohol = st.sidebar.selectbox("Trinken Sie regelmässig?", options=["Ja", "Nein"])
geschlecht = st.sidebar.selectbox("Geschlecht", options=["Männlich", "Weiblich"])
physicalHealth = st.sidebar.slider("Wie oft in den vergangen 30 Tagen füllten Sie sich physisch nicht gut?", 0, 30, 5)
mentalHealth = st.sidebar.slider("Wie oft in den vergangen 30 Tagen füllten Sie sich mental nicht gut?", 0, 30, 5)
schlaganfall = st.sidebar.selectbox("Schlaganfall", options=["Ja", "Nein"])
problemeBeimGehen = st.sidebar.selectbox("Haben Sie Probleme beim Gehen?", options=["Ja", "Nein"])
asthma = st.sidebar.selectbox("Asthma", options=["Ja", "Nein"])
physicalactivity = st.sidebar.selectbox("Physical Activity", options=["Ja", "Nein"])
kidneyDisease = st.sidebar.selectbox("Kidney Disease", options=["Ja", "Nein"])
skinCancer = st.sidebar.selectbox("Skin Cancer", options=["Ja", "Nein"])
diabetic = st.sidebar.selectbox("Diabetic", options=["Ja", "Nein"])
genHealth = st.sidebar.selectbox("GenHealth", options=["Poor", "Fair", "Excellent", "Good", "Very Good"])
ageCategory = st.sidebar.selectbox("Age Category", options=["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"])
rasse = st.sidebar.selectbox("Race", options=["American Indian/Alaskan Native", "Asian", "Black", "Hispanic", "Other", "White"])


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


st.write(berechneHeartDisease())