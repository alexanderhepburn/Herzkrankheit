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

    def alterKonvertieren(self, x):
        switch = {
            "18-24": 0,
            "25-29": 1,
            "30-34": 2,
            "35-39": 3,
            "40-44": 4,
            "45-49": 5,
            "50-54": 6,
            "55-59": 7,
            "60-64": 8,
            "65-69": 9,
            "70-74": 10,
            "75-79": 11,
            "80+": 12
        }
        return switch.get(x, "Error")

    def rasseKonvertieren(self, x):
        switch = {
            "American Indian/Alaskan Native": 0,
            "Asian": 1,
            "Black": 2,
            "Hispanic": 3,
            "Other": 4,
            "White": 5
        }
        return switch.get(x, "Error")

    def jaOderNein(self, input):
        if input == "Ja":
            return 1
        else:
            return 0

    def sexKonvertieren(self, x):
        if x == "Männlich":
            return 1
        else:
            return 0

    def genHealthKonvertieren(self, x):
        switch = {
            'Poor': 0,
            'Fair': 1,
            'Excellent': 2,
            'Good': 3,
            'Very Good': 4
        }
        return switch.get(x, "Error")

    def berechneHeartDisease(self, x, model):
        data = {
            "BMI": [x.körpergewicht / ((x.körpergrösse / 100) ** 2)],
            'Smoking': [self.jaOderNein(x.raucher)],
            'AlcoholDrinking': [self.jaOderNein(x.alkohol)],
            "Stroke": [self.jaOderNein(x.schlaganfall)],
            "PhysicalHealth": [x.physicalHealth],
            "MentalHealth": [x.mentalHealth],
            "DiffWalking": [self.jaOderNein(x.problemeBeimGehen)],
            "Sex": [self.sexKonvertieren(x.geschlecht)],
            "AgeCategory": [self.alterKonvertieren(x.ageCategory)],
            "Race": [self.rasseKonvertieren(x.rasse)],
            "Diabetic": [self.jaOderNein(x.diabetic)],
            "PhysicalActivity": [self.jaOderNein(x.physicalactivity)],
            "GenHealth": [self.genHealthKonvertieren(x.genHealth)],
            "SleepTime": [x.schlaffZeit],
            "Asthma": [self.jaOderNein(x.asthma)],
            "KidneyDisease": [self.jaOderNein(x.kidneyDisease)],
            "SkinCancer": [self.jaOderNein(x.skinCancer)]
        }
        inputInfos = pd.DataFrame(data=data)
        heartdisease = model.predict_proba(inputInfos)[0][1]
        return heartdisease


