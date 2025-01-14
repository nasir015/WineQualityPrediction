import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.WineQualityPrediction.utils.my_logging import logger
from src.WineQualityPrediction.utils.my_exception import CustomException
from src.WineQualityPrediction.config.configuration import ConfigurationManager
from src.WineQualityPrediction.components.data_validation import DataValidation
import logging
log_path = 'log\log_file.log'

Stage_name = "Data Validation Stage"

class DataValidationTrainingPipeline:

    def __init__(self):
        pass

    def initiate_data_validation(self):
        try:
            logger(log_path, logging.INFO, "Data validation script started....")

            # Step 1: Configuration setup
            logger(log_path, logging.INFO, "Initializing ConfigurationManager....")
            config_manager = ConfigurationManager()

            # Step 2: Fetching data validation configuration
            logger(log_path, logging.INFO, "Fetching data validation configuration....")
            data_validation_config = config_manager.get_data_validation_config()

            # Step 3: Performing data validation
            logger(log_path, logging.INFO, "Initializing DataValidation....")
            data_validation = DataValidation(config=data_validation_config)

            # Step 4: Running data validation
            logger(log_path, logging.INFO, "Running data validation....")
            validation_status = data_validation.validate_all_columns()

            if validation_status:
                logger(log_path, logging.INFO, "Data validation successful....")
            else:
                logger(log_path, logging.ERROR, "Data validation failed.....")

        except CustomException as custom_ex:
            logger(log_path, logging.ERROR, f"An unexpected error occurred: {CustomException(custom_ex,sys)}")
            raise CustomException(custom_ex,sys)  # Optionally, re-raise the exception for external handling
        except Exception as ex:
            logger(log_path, logging.ERROR, f"An unexpected error occurred: {CustomException(ex,sys)}")
            raise CustomException(ex,sys)  # Optionally, re-raise the exception for external handling


if __name__ == "__main__":
    try:
        logger(log_path, logging.INFO, f">>>>>>>>>>Starting {Stage_name}<<<<<<<<<<<")
        pipeline = DataValidationTrainingPipeline()
        pipeline.initiate_data_validation()
        logger(log_path, logging.INFO, f">>>>>>>>>>{Stage_name} completed successfully.<<<<<<<<<<<")
    except CustomException as ce:
        logger(log_path, logging.ERROR, f"CustomException occurred: {CustomException(ce, sys)}")
        raise CustomException(ce, sys)