import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from Herz import Herz

#@st.cache()

x_train = pd.read_csv("x_train_heart_data.csv")
x_test  = pd.read_csv("x_test_heart_data.csv")
x_train = pd.read_csv("x_train_heart_data.csv")
Log_Reg = pickle.load(open('finalized_LogReg_model.sav', 'rb'))
################
################

manager = Herz()

#UI

st.set_page_config(
    page_title = 'Heart Health Assessment App',
    page_icon = '❤️',
    layout = 'wide'
    )
st.title('Heart Health Assessment App')

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
model = LogisticRegression(random_state=1, C=0.004291934260128779, class_weight='balanced', dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=3000, multi_class='auto', n_jobs=None, penalty='l2',  solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
Log_Reg = model.fit(x_train, y_train)

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
    heartdisease = model.predict_proba(inputInfos)[0][1]
    return heartdisease


st.write(berechneHeartDisease())