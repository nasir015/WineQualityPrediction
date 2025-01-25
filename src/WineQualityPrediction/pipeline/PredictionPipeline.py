import pickle
from pathlib import Path
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        self.model=pickle.load(open('artifacts/model_trainer/best_model.pkl', 'rb'))

    def predict(self,data):
        prediction=self.model.predict(data)

        return prediction

