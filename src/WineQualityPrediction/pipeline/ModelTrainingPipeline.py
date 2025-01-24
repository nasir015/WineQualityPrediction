import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.WineQualityPrediction.utils.my_logging import logger
from src.WineQualityPrediction.utils.my_exception import CustomException
from src.WineQualityPrediction.config.configuration import ConfigurationManager
from src.WineQualityPrediction.components.model_training import ModelTrainer
import logging

log_path = 'log\log_file.log'
Stage_name = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
        try:
            logger(log_path, logging.INFO, "Model Training script started....")

            # Step 1: Configuration setup
            logger(log_path, logging.INFO, "Initializing ConfigurationManager....")
            config_manager = ConfigurationManager()

            # Step 2: Fetching model training configuration
            logger(log_path, logging.INFO, "Fetching model training configuration....")
            model_trainer_config = config_manager.get_model_trainer_config()

            # Step 3: Performing model training
            logger(log_path, logging.INFO, "Initializing Model Trainer....")
            model_trainer = ModelTrainer(config=model_trainer_config)

            # Step 4: Running Model Training
            logger(log_path, logging.INFO, "Running Model Training....")
            try:
                model_trainer.objective()
            except Exception as ex:
                logger(log_path, logging.ERROR, f"An unexpected error occurred: {CustomException(ex,sys)}")
                raise CustomException(ex,sys)
            
            logger(log_path,logging.INFO, 'Best model training strated....')
            try:
                model_trainer.train_best_model()
            except Exception as ex:
                logger(log_path, logging.ERROR, f"An unexpected error occurred: {CustomException(ex,sys)}")
                raise CustomException(ex,sys)


            
        except Exception as ex:
            logger(log_path, logging.ERROR, f"An unexpected error occurred: {CustomException(ex,sys)}")
            raise CustomException(ex,sys)  # Optionally, re-raise the exception for external handling

if __name__ == "__main__":
    model_training_pipeline = ModelTrainingPipeline()
    model_training_pipeline.initiate_model_training()