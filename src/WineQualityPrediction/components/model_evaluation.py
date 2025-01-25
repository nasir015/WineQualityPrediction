import mlflow.artifacts
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from src.WineQualityPrediction.utils.my_logging import logger
from src.WineQualityPrediction.utils.my_exception import CustomException
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report
from src.WineQualityPrediction.utils.common import *
from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.WineQualityPrediction.entity.config_entity import ModelEvaluationConfig
from mlflow.models.signature import infer_signature
from sklearn.exceptions import UndefinedMetricWarning
import warnings
log_path = 'log\log_file.log'

class ModelEvaluation:
    """
    Initializes the ModelEvaluation class with a given configuration.

    Parameters:
        config (ModelEvaluationConfig): An object containing configuration parameters 
                                        for model evaluation, including paths for data, 
                                        model, MLflow URI, and metric file storage.
    """    
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    
    
    
    def log_into_mlflow(self):
        """
        Logs evaluation metrics and model details into MLflow and optionally registers the model.
        This method sets up MLflow tracking and authenticates using environment variables.
        """
        # Set MLflow tracking environment variables
        os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/nasir.uddin.6314/WineQualityPrediction_with_Docker.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = "nasir.uddin.6314"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "026054fecb7c525dea65edc373c6935c9a1332fc"

        # Configure MLflow registry URI
        mlflow.set_registry_uri(os.getenv("MLFLOW_TRACKING_URI"))
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Load pkl model
        model = load_model(self.config.model_path, self.config.model_name)

        # Split test data into features (X) and target (Y)
        test_x = pd.read_csv(self.config.X_test)
        test_y = pd.read_csv(self.config.y_test)
        test_y = test_y.squeeze()

        scale = StandardScaler()

        test_x = scale.fit_transform(test_x)

        # predict target values using the loaded model
        predicted_qualities = model.predict(test_x)
        # Log model evaluation metrics
        accuracy = accuracy_score(test_y, predicted_qualities)
        # Start an MLflow run
        with mlflow.start_run(run_name="ModelEvaluationRun"):
            # Predict target values using the loaded model
            # predict target values using the loaded model
            predicted_qualities = model.predict(test_x)
            # Log model evaluation metrics
            accuracy = accuracy_score(test_y, predicted_qualities)
            
            mlflow.log_metrics({"accuracy": accuracy})
            report = classification_report(test_y, predicted_qualities)

            # Log the classification report
            mlflow.log_text(report,artifact_file="classification_report.txt")

            input_example = test_x
            signature = infer_signature(test_x, predicted_qualities)
            # Log and optionally register the model
            if tracking_url_type_store != "file":
                # Register the model if the tracking store supports it
                mlflow.sklearn.log_model(model, "model", registered_model_name="WineQualityPredictionModel",
                                         signature=signature, input_example=input_example)
            else:
                # Log the model without registering
                mlflow.sklearn.log_model(model, "model")

            print("Model evaluation metrics and model details logged successfully.")

