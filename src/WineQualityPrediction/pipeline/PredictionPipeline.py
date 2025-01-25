import pickle
from pathlib import Path


class PredictionPipeline:
    def __init__(self):
        self.model=pickle.load(open('artifacts/model_trainer/best_model.pkl', 'rb'))

    def predict(self,data):
        prediction=self.model.predict(data)

        return prediction