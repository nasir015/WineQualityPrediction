import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.WineQualityPrediction.utils.my_logging import logger
from src.WineQualityPrediction.utils.my_exception import CustomException
from src.WineQualityPrediction.config.configuration import ConfigurationManager
from  src.WineQualityPrediction.components.data_transformation import DataTransformation
import logging

log_path = 'log\log_file.log'

Stage_name = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        try:
            # Initialize configuration manager
            config = ConfigurationManager()
            logger(log_path, logging.INFO, "ConfigurationManager initialized successfully.")
            
            # Get data transformation configuration
            data_transformation = config.get_data_transformation_config()
            logger(log_path, logging.INFO, "Data transformation configuration fetched successfully.")

            transformation = DataTransformation(data_transformation)
            logger(log_path, logging.INFO, "DataTransformation class initialized successfully.")

            transformation.train_test_spliting()
            logger(log_path, logging.INFO, "Data split successfully.")
        except Exception as e:
            logger(log_path, logging.ERROR, f"An unexpected error occurred: {CustomException(e, sys)}")
            raise CustomException(e, sys)


