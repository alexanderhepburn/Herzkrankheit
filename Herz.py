import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class Herz:
    """Herz Class"""
    def __init__(self):
        print("init")

    def speicherModell(self):
        df = pd.read_csv("heart_2020_cleaned.csv")
        xTrain, xTest, yTrain, yTest = train_test_split(df["HeartDisease"], df.drop(axis=1, columns=["HeartDisease"]),
                                                        random_state=1)

        model = LogisticRegression()
        model.fit(xTrain, yTrain)

        pickle.dump(model, open("LogisticRegression.sav", 'wb'))
        print("Gespeichert!")



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from Herz import Herz
import streamlit as st

#MARK: Init Config
st.set_page_config(page_title="Herz", layout="wide")



st.title("Herzkrankheit")
df = pd.read_csv("heart_2020_cleaned.csv")
person = df.drop(axis=1, columns=["HeartDisease"])
person = person.iloc[0]

#loaded_model = pickle.load(open("LogisticRegression.sav", 'rb'))
#loaded_model.predict(person)


st.sidebar.title("Input deine Infos!")
st.sidebar.slider("Gewicht", min_value=20, max_value=200, value=60)
st.sidebar.slider("Körper Grösse", min_value=20, max_value=200, value=60)

