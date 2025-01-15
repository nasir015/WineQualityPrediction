import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from src.WineQualityPrediction.utils.my_logging import logger
from src.WineQualityPrediction.utils.my_exception import CustomException
from sklearn.linear_model import ElasticNet
import joblib
from src.WineQualityPrediction.entity.config_entity import ModelTrainerConfig

log_path = 'log\log_file.log'

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer class with a given configuration.

        Parameters:
            config (ModelTrainerConfig): An object containing the configuration 
                                        parameters for training, including paths 
                                        to datasets, model hyperparameters, and 
                                        save paths.
        """
        self.config = config

    def train(self):
        logger(log_path,logging.INFO, "Training the model Started...")

        """
        Trains the ElasticNet regression model using the provided configuration.

        Steps:
            1. Reads training and testing datasets from the specified paths.
            2. Separates features (X) and target (Y) from the datasets.
            3. Initializes and trains the ElasticNet model with specified hyperparameters.
            4. Saves the trained model to the configured directory.

        Dependencies:
            - pandas (pd): Used to read the CSV datasets.
            - sklearn.linear_model.ElasticNet: The regression model used for training.
            - joblib: Used for saving the trained model to disk.
            - os: Handles file paths for saving the model.

        Raises:
            FileNotFoundError: If the training or testing data files are not found.
            ValueError: If the target column is not present in the datasets.

        Notes:
            Ensure that the `train_data_path` and `test_data_path` in the configuration point to valid CSV files.
            The directory specified by `root_dir` should exist or be created beforehand.
        """     
        logger(log_path,logging.INFO, "Reading the datasets...")   
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        logger(log_path,logging.INFO, "Separating features and target...")
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        logger(log_path,logging.INFO, "strart the model building... ")
        logger(log_path,logging.INFO, "Training the model...")
        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        logger(log_path,logging.INFO, "Model training completed...")
        logger(log_path,logging.INFO, "Saving the model...")
        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))
        logger(log_path,logging.INFO, "Model saved...")