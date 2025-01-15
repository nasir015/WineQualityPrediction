import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.WineQualityPrediction.utils.my_logging import logger
from src.WineQualityPrediction.utils.my_exception import CustomException
from src.WineQualityPrediction.config.configuration import ConfigurationManager
from src.WineQualityPrediction.components.model_evaluation import ModelEvaluation
import logging

log_path = 'log\log_file.log'


class ModelEvaluationPipelines:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):  # Added 'self' as the first parameter
        try:
            logger(log_path, logging.INFO, "Model Evaluation script started....")

            # Step 1: Configuration setup
            logger(log_path, logging.INFO, "Initializing ConfigurationManager....")
            config_manager = ConfigurationManager()

            # Step 2: Fetching model evaluation configuration
            logger(log_path, logging.INFO, "Fetching model evaluation configuration....")
            model_evaluation_config = config_manager.get_model_evaluation_config()

            # Step 3: Performing model evaluation
            logger(log_path, logging.INFO, "Initializing Model Evaluation....")
            model_evaluation = ModelEvaluation(config=model_evaluation_config)

            # Step 4: Running Model Evaluation
            logger(log_path, logging.INFO, "Running Model Evaluation....")
            Eval_status = model_evaluation.log_into_mlflow()

            if Eval_status:
                logger(log_path, logging.INFO, "Model Evaluation successful....")
            else:
                logger(log_path, logging.ERROR, "Model Evaluation failed.....")

        except CustomException as custom_ex:
            logger(log_path, logging.ERROR, f"An unexpected error occurred: {CustomException(custom_ex, sys)}")
            raise CustomException(custom_ex, sys)  # Optionally, re-raise the exception for external handling
        except Exception as ex:
            logger(log_path, logging.ERROR, f"An unexpected error occurred: {CustomException(ex, sys)}")
            raise CustomException(ex, sys)  # Optionally, re-raise the exception for external handling
