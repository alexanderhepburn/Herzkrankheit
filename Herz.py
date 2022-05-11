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



