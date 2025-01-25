import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.WineQualityPrediction.utils.my_logging import logger
from src.WineQualityPrediction.utils.my_exception import CustomException
from src.WineQualityPrediction.config.configuration import ConfigurationManager
from src.WineQualityPrediction.components.data_ingestion import DataIngestion
import logging
import sys
log_path = 'log\log_file.log'

Stage_name = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    def initiate_data_ingestion(self):
        try:
            # Initialize configuration manager
            config = ConfigurationManager()
            logger(log_path, logging.INFO, "ConfigurationManager initialized successfully.")
            
            # Get data ingestion configuration
            data_ingestion_config = config.get_data_ingestion_config()
            logger(log_path, logging.INFO, "Data ingestion configuration fetched successfully.")
            
            # Initialize data ingestion
            data_ingestion = DataIngestion(config=data_ingestion_config)
            logger(log_path, logging.INFO, "DataIngestion instance created successfully.")
            
            # Download the data file
            data_ingestion.download_file()
            logger(log_path, logging.INFO, "File downloaded successfully.")
            
            # Extract the downloaded zip file
            data_ingestion.extract_zip_file()
            logger(log_path, logging.INFO, "Zip file extracted successfully.")
            
        except CustomException as ce:
            logger(log_path, logging.ERROR, f"CustomException occurred: {CustomException(ce, sys)}")
            raise CustomException(ce, sys)
        except Exception as e:
            logger(log_path, logging.ERROR, f"An unexpected error occurred: {CustomException(e, sys)}")
            raise CustomException(e, sys)


