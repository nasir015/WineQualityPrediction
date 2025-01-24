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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.WineQualityPrediction.utils.common import *
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from src.WineQualityPrediction.entity.config_entity import ModelEvaluationConfig

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

    
    
    
    def eval_metrics(self,actual, pred):
        """
        Calculates evaluation metrics for the model predictions.

        Parameters:
            actual (array-like): Ground truth target values.
            pred (array-like): Predicted target values from the model.

        Returns:
            tuple: A tuple containing:
                - `rmse` (float): Root Mean Squared Error.
                - `mae` (float): Mean Absolute Error.
                - `r2` (float): R-squared score.

        Dependencies:
            - numpy: Used for mathematical calculations (e.g., square root).
            - sklearn.metrics: For calculating regression metrics.
        """        
        
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    
    
    
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

        # Load test data and model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        # Split test data into features (X) and target (Y)
        test_x = test_data.drop(columns=[self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        # Start an MLflow run
        with mlflow.start_run(run_name="ModelEvaluationRun"):
            # Predict target values using the loaded model
            predicted_qualities = model.predict(test_x)

            # Evaluate metrics
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

            # Save evaluation metrics locally
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            # save this score as a json file 
            save_json(self.config.metric_file_name, scores)

            # Log parameters and metrics into MLflow
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Log and optionally register the model
            if tracking_url_type_store != "file":
                # Register the model if the tracking store supports it
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
            else:
                # Log the model without registering
                mlflow.sklearn.log_model(model, "model")

            print("Model evaluation metrics and model details logged successfully.")